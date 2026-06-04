# EXPLANATION — 2026-06-04

## Obiettivo del progetto

Addestrare una policy di get-up per il robot Unitree G1 a 29 gradi di liberta in simulazione MuJoCo, con l'obiettivo finale di trasferire la policy sul robot fisico (sim2real). Il robot deve imparare ad alzarsi dal suolo partendo da una posa supina (schiena a terra) o prona (faccia a terra), in qualsiasi orientazione planare.

Task ID: `Unitree-G1-GetUp`. Entry point: `bash train.sh Unitree-G1-GetUp`. Framework: mjlab (wrapper MuJoCo), algoritmo PPO via rsl_rl. Dual hardware: AMD/CPU/osmesa per sviluppo, NVIDIA/GPU/EGL per training.

---

## Cosa e stato implementato nel blocco P1

Il blocco P1 ha aggiunto cinque feature distinte: un nuovo reward per la fase iniziale del rialzarsi (P1.1), un intero modulo di metriche diagnostiche per stage/successo/terminazione/saturazione (P1.2 e P1.3), tre forme di domain randomization pronte ma disabilitate (P1.4, P1.5, P1.6), e un sistema completo di contact logging per parti del corpo (descritto nella sezione dedicata).

---

## P1.1 — Reward `prone_supine_righting`

**File:** `src/tasks/getup/mdp/rewards.py`

Aggiunta la funzione `prone_supine_righting`, che incentiva il robot a lasciare le pose piatte (supino/prono) durante la fase iniziale del rialzarsi.

Il segnale di orientamento e identico a `body_up_exp`: `exp(-projected_gravity_b[:, 2])`, che raggiunge il massimo quando il body Z del pelvis punta verso l'alto (robot eretto) e il minimo quando il robot e capovolto. La differenza rispetto a `body_up_exp` e il moltiplicatore di gating:

```python
early_gate = 1.0 - _standing_gate(h, threshold=0.40, band=0.10)
```

La funzione `_standing_gate` (gia esistente) produce un valore continuo 0 → 1 man mano che il pelvis supera la soglia indicata. Moltiplicando per `1 - gate`, il reward `prone_supine_righting` e attivo (~1.0) quando il robot e a terra e si azzera gradualmente man mano che il pelvis sale verso 0.40 m. Sopra quella soglia il reward e spento.

La motivazione del gating e fondamentale: un reward di orientamento sempre attivo contrasterebbe le posture intermedie che il robot assume nelle fasi di transizione (inginocchiato, piegato in avanti), dove il pelvis non punta ancora verso l'alto ma il robot sta comunque progredendo verso la postura eretta. Con il gate, il reward agisce solo nella fase 0/1 dove il robot deve scegliere da che parte ruotare per iniziare a sollevarsi.

Peso assegnato in `getup_env_cfg.py`: 0.5.

---

## P1.2 e P1.3 — Modulo metriche diagnostiche

**File:** `src/tasks/getup/mdp/metrics.py` (file nuovo, poi esteso con il contact logging)

E stato creato un modulo da zero con 23 funzioni metriche, tutte con firma `(env, ...) -> Tensor[B,]`. Il `MetricsManager` di mjlab accumula la somma per episodio e pubblica la media per-episodio in `extras["log"]`. Le chiavi che contengono "/" vengono forwarded automaticamente al logger tensorboard/wandb di rsl_rl.

### Stage occupancy (P1.2)

Quattro funzioni classificano ogni env in una delle quattro fasce di altezza pelvis:

| Chiave | Intervallo pelvis | Significato |
|--------|-------------------|-------------|
| `stage/stage0` | < 0.20 m | Robot a terra, righting |
| `stage/stage1` | 0.20 – 0.40 m | Transizione supporto |
| `stage/stage2` | 0.40 – 0.65 m | Salita attiva |
| `stage/stage3` | > 0.65 m | In piedi |

Monitorare l'evoluzione di queste metriche nel tempo e il modo piu diretto per leggere il curriculum implicito: se `stage/stage3` cresce nel corso del training, la policy sta imparando a rialzarsi.

### Success metrics (P1.2)

- `success/candidate`: robot sopra 0.65 m (potenzialmente in piedi) — misura la frequenza di "toccare" la soglia
- `success/stable_hold`: `standing_counter >= 50` step consecutivi (0.5 s a 100 Hz) — indica che il robot regge la postura, non solo la sfiora
- `success/ever_stood`: il robot ha raggiunto h > 0.65 m almeno una volta nell'episodio — misura il tasso di episodi con successo
- `success/fall_after_success`: il robot aveva gia raggiunto la postura eretta ma ora e sotto soglia — identifica episodi in cui il robot si rialza e poi ricade

### Termination reasons (P1.2)

- `termination/timeout`: l'episodio e scaduto regolarmente
- `termination/failure`: l'episodio e terminato per condizione di fallimento (NaN, esplosione, ground penetration, standing_fall_timeout)
- `termination/fall_after_success`: subset di failure in cui `ever_stood == True` — distingue "mai riuscito" da "riuscito e poi caduto"

Queste tre metriche, mediate su episodi, consentono di leggere la distribuzione delle cause di terminazione durante il training.

### Action metrics (P1.3)

- `action/norm_mean`: norma L2 dell'azione grezza
- `action/saturation_fraction`: frazione di DOF con `|a| > 0.95` — proxy per policy che "sbatte" sui limiti del range di azione normalizzato
- `action/rate_mean`: norma del delta azione step-to-step
- `action/acc_mean`: norma dell'accelerazione dell'azione (differenza seconda)

### Torque metrics (P1.3)

- `torque/norm_mean`: norma L2 delle forze attuatore `asset.data.actuator_force`
- `torque/saturation_fraction`: frazione di attuatori con `|tau| / tau_limit > 0.90` — saturazione fisica del motore
- `torque/power_mean`: potenza meccanica media `|tau * qdot|` — proxy di consumo energetico e di comportamenti aggressivi

Per le torque metrics, la funzione interna `_get_effort_limits` gestisce correttamente sia la forma con DR attiva (`forcerange` shape `[n_envs, n_actuators, 2]`) sia quella senza DR (`[n_actuators, 2]`).

### Stage-aware saturation (P1.3)

Quattro metriche incrociate `stage0/action_saturation`, `stage3/action_saturation`, `stage0/torque_saturation`, `stage3/torque_saturation` moltiplicano la saturation fraction per una maschera binaria di stage. Questo consente di distinguere se la saturazione avviene quando il robot e a terra (stage 0, puo essere accettabile nelle fasi esplorative) o quando e in piedi (stage 3, potenzialmente problematico per il sim2real).

---

## Contact logging — sensori di contatto per parti del corpo

**File modificati:**
- `src/tasks/getup/config/g1/env_cfgs.py` — definizione di 8 nuovi `ContactSensorCfg`
- `src/tasks/getup/mdp/metrics.py` — 2 helper generici + 15 nuovi `MetricsTermCfg`

### Motivazione

Sapere se il robot tocca il suolo con testa, torso, ginocchia, avambracci o mani e informazione diagnostica essenziale durante il training del get-up. Senza questi dati non e possibile rispondere a domande fondamentali quali: "il robot usa le mani per spingersi?", "il torso tocca terra per troppo tempo?", "le ginocchia vengono caricate in modo anomalo durante la transizione?". Il contact logging consente di correlare il comportamento emergente della policy con le fasi cinematiche del movimento.

### Architettura dei sensori

Tutti e 8 i nuovi sensori usano la stessa configurazione strutturale:

```
mode     = "geom"     — target su geom specifiche (vs "subtree" che includerebbe geom non pertinenti)
reduce   = "maxforce" — per ogni primary geom, conserva il contatto con forza massima
secondary = terrain   — rileva solo il contatto con il pavimento
fields   = ("found", "force")
num_slots = 1
```

La scelta di `mode="geom"` e la piu importante: la modalita `subtree` avrebbe catturato tutti i geom figli di un link, inclusi geom interni e link di transizione, producendo falsi positivi. Con `mode="geom"` il match e chirurgico sul nome del geom di collisione esatto.

La scelta di `reduce="maxforce"` (invece di `netforce` usato per i piedi) e motivata dall'obiettivo: per il logging diagnostico di parti del corpo non portanti e sufficiente conoscere se c'e contatto e con che forza dominante, senza bisogno della forza risultante netta.

**Shape dei tensori dopo il build:**
- `found`: `[B, N_primaries]` — 0 = nessun contatto, >0 = contatto rilevato
- `force`: `[B, N_primaries, 3]` — vettore forza del contatto piu forte per ogni primary

### Inventario sensori

| Sensore | Geom primari | Corpo anatomico |
|---------|-------------|-----------------|
| `contact_head` | `head_collision` (1 geom) | Testa |
| `contact_torso` | `torso_collision` + `pelvis_collision` (2 geom) | Torso e bacino |
| `contact_knee_left` | `left_shin_collision` + `left_linkage_brace_collision` (2 geom) | Ginocchio sinistro |
| `contact_knee_right` | `right_shin_collision` + `right_linkage_brace_collision` (2 geom) | Ginocchio destro |
| `contact_forearm_left` | `left_elbow_yaw_collision` + `left_wrist_collision` (2 geom) | Avambraccio sinistro |
| `contact_forearm_right` | `right_elbow_yaw_collision` + `right_wrist_collision` (2 geom) | Avambraccio destro |
| `contact_hand_left` | `left_hand_collision` (1 geom) | Mano sinistra |
| `contact_hand_right` | `right_hand_collision` (1 geom) | Mano destra |

I piedi riutilizzano il sensore gia esistente `feet_ground_contact` (modalita `subtree`, `reduce="netforce"`) per evitare duplicazione.

Le ginocchia e gli avambracci usano 2 geom ciascuno per coprire l'intera superficie anatomica: per il ginocchio la capsula della tibia (`shin`) e la capsula del rinforzo cinematico (`linkage_brace`); per l'avambraccio la capsula del gomito (`elbow_yaw`) e quella del polso (`wrist`).

### Helper generici in metrics.py

Due funzioni generiche gestiscono entrambi i pattern di accesso al tensore del sensore:

```python
def contact_found(env, sensor_name) -> Tensor[B,]:
    sensor = env.scene[sensor_name]
    found = sensor.data.found          # [B, N_primaries]
    return (found > 0).any(dim=1).float()
```

`any(dim=1)` aggrega correttamente sia sensori a 1 primary (head, mani) sia sensori a 2 primary (torso, ginocchia, avambracci): il risultato e 1 se almeno un geom del gruppo e in contatto.

```python
def contact_force_max(env, sensor_name) -> Tensor[B,]:
    sensor = env.scene[sensor_name]
    force = sensor.data.force          # [B, N_primaries, 3]
    mag = torch.norm(force, dim=-1)    # [B, N_primaries]
    return mag.max(dim=1).values       # [B,]
```

La forza massima tra i primary e la metrica piu interpretabile: per il torso, per esempio, riporta la forza del geom (pelvis o torso) con il contatto piu pesante, che e il valore rilevante per valutare l'impatto.

Entrambe le funzioni gestiscono il caso `None` (sensore non ancora inizializzato o nessun dato disponibile) restituendo un tensore di zeri, evitando crash durante i primi step.

### 15 nuovi MetricsTermCfg — chiavi di log

Tutte le metriche sono pubblicate sotto il prefisso `Episode_Metrics/contact/...`:

| Chiave log | Helper | Sensore |
|------------|--------|---------|
| `contact/head` | `contact_found` | `contact_head` |
| `contact/head_force_max` | `contact_force_max` | `contact_head` |
| `contact/torso` | `contact_found` | `contact_torso` |
| `contact/torso_force_max` | `contact_force_max` | `contact_torso` |
| `contact/knee_left` | `contact_found` | `contact_knee_left` |
| `contact/knee_right` | `contact_found` | `contact_knee_right` |
| `contact/forearm_left` | `contact_found` | `contact_forearm_left` |
| `contact/forearm_right` | `contact_found` | `contact_forearm_right` |
| `contact/forearm_left_force_max` | `contact_force_max` | `contact_forearm_left` |
| `contact/forearm_right_force_max` | `contact_force_max` | `contact_forearm_right` |
| `contact/hand_left` | `contact_found` | `contact_hand_left` |
| `contact/hand_right` | `contact_found` | `contact_hand_right` |
| `contact/hand_left_force_max` | `contact_force_max` | `contact_hand_left` |
| `contact/hand_right_force_max` | `contact_force_max` | `contact_hand_right` |
| `contact/feet` | `contact_found` | `feet_ground_contact` |

Nota: per ginocchia e piedi vengono loggate solo le metriche binarie (nessuna `_force_max`), in quanto l'informazione di forza e gia coperta dal sensore piedi esistente e le ginocchia non sono parti portanti primarie nel get-up.

### Parametri MuJoCo aggiornati

Due parametri di simulazione sono stati aumentati per supportare il numero maggiore di contatti generati dal get-up con tutti i sensori attivi:

- `cfg.sim.nconmax`: 48 → 100 (massimo contatti simultanei nel solver)
- `cfg.sim.contact_sensor_maxmatch`: 200 → 400 (massimo match per il sistema di risoluzione dei sensori)

Il get-up e una delle task con piu contatti simultanei nel ciclo di vita di un episodio: nella fase supina il torso, il pelvis, la testa e le braccia possono essere tutti a terra contemporaneamente. Il valore precedente di `nconmax=48` era dimensionato per il walking e risultava insufficiente.

### Risultati di validazione

Test eseguito su 4 env, 20 step, policy zero (azioni nulle → robot immobile nella posa iniziale campionata casualmente):

| Metrica | Valore osservato | Interpretazione |
|---------|-----------------|-----------------|
| `contact/torso` | 75% rate, ~180 N peak | Robot supino: torso/pelvis a terra, atteso |
| `contact/head` | 50% rate, ~22 N | Alcune pose iniziali hanno la testa a terra |
| `contact/hand_left`, `contact/hand_right` | 50% rate, 17–20 N | Pose prone/laterali con mani a terra |
| `contact/knee_left`, `contact/knee_right` | 0% | Ginocchia sollevate nella posa supina piatta |
| Totale chiavi nel log | 38 | Include metriche pre-esistenti + 15 nuove |

Il risultato delle ginocchia a 0% e atteso: la posa supina standard ha le gambe distese con le ginocchia a circa 0.15 m da terra. Il contatto con le ginocchia emergera nelle fasi di transizione quando il robot inizia a piegarle per alzarsi.

### Scelte di design non ovvie

**Perche non loggare posizione e normale del contatto.** La posizione e la normale del punto di contatto avrebbero fornito informazioni cinematiche precise (dove esattamente il robot tocca terra), ma avrebbero richiesto tensori `[B, N_primaries, 3]` aggiuntivi per ogni sensore, aumentando l'uso di memoria del 60–80% per i sensori di contatto. Per metriche di training diagnostico, la coppia (binario + forza massima) e sufficiente a rispondere alle domande operative.

**Perche `reduce="maxforce"` invece di `reduce="netforce"` per le parti del corpo.** La forza netta e appropriata per i piedi perche si vuole la forza totale di supporto del peso. Per torso, testa e arti la forza dominante di un singolo geom e la grandezza piu interpretabile: rappresenta il picco di impatto, non la somma di contatti multipli sovrapposti che potrebbero essere artefatti del solver.

**Perche il sensore torso ha sia `torso_collision` che `pelvis_collision`.** I due geom sono capsule distinte (torso = capsula lungo la colonna, pelvis = sfera all'anca) che appartengono a link diversi ma anatomicamente rappresentano la stessa regione funzionale "schiena/fianchi". Raggrupparli in un unico sensore produce una metrica che risponde alla domanda "la parte centrale del corpo tocca terra?" senza richiedere due metriche separate da aggregare manualmente in post-processing.

---

## P1.4 — Domain Randomization: motor strength

**File:** `src/tasks/getup/config/g1/env_cfgs.py`

Aggiunto un evento `motor_strength` di tipo `reset` che usa `dr.effort_limits(operation="scale")` per scalare uniformemente i limiti di forza degli attuatori nell'intervallo [0.90, 1.10] ad ogni reset di ogni env. Simula la variabilita nella forza effettiva dei motori tra robot fisici diversi o per usura.

Disabilitato di default (`_DR_MOTOR_STRENGTH_ENABLE = False`). L'intero blocco e in un `if`, costo zero a runtime quando il flag e False.

---

## P1.5 — Domain Randomization: PD gains

**File:** `src/tasks/getup/config/g1/env_cfgs.py`

Aggiunto un evento `pd_gains` di tipo `reset` che usa `dr.pd_gains(operation="scale")` per scalare Kp e Kd nell'intervallo [0.85, 1.15] per env ad ogni reset. L'implementazione di mjlab modifica direttamente `actuator_gainprm` e `actuator_biasprm` nel modello MuJoCo per env, che sono i tensori che determinano rispettivamente il guadagno proporzionale e derivativo del controllo.

Disabilitato di default (`_DR_PD_GAINS_ENABLE = False`).

---

## P1.6 — Domain Randomization: action delay

**File:** `src/assets/robots/unitree_g1/g1_constants.py` + `src/tasks/getup/config/g1/env_cfgs.py`

Questa e la DR piu strutturalmente complessa del blocco P1 perche richiede una modifica alla configurazione dell'attuatore, non solo ai parametri del modello.

**In `g1_constants.py`:** sono stati aggiunti tre elementi:

1. La funzione helper `_wrap_with_delay(base_cfg)` che wrappa un `BuiltinPositionActuatorCfg` in un `DelayedActuatorCfg`, con buffer di lag 0–3 steps.
2. `G1_ARTICULATION_DELAYED`: variante di `G1_ARTICULATION` dove tutti e 6 i gruppi attuatori sono wrapped con `DelayedActuatorCfg`.
3. La funzione `get_g1_supine_robot_cfg_with_delay()`: restituisce la config supine usando `G1_ARTICULATION_DELAYED` invece di `G1_ARTICULATION`.

**In `env_cfgs.py`:** aggiunto un evento `action_delay` di tipo `reset` che usa `dr.sync_actuator_delays(lag_range=(0, 3))` per assegnare un lag casuale (in physics steps) ad ogni env ad ogni reset. Con timestep 0.002 s, `lag_range=(0, 3)` corrisponde a 0–6 ms di ritardo, che e l'intervallo realistico per l'hardware G1.

Il delay non viene aggiunto alle osservazioni della policy: e un disturbo di attuazione implicito, e la policy deve imparare ad essere robusta senza conoscere il lag corrente.

**Attenzione alla doppia attivazione:** per usare questa DR occorrono due modifiche coordinate: sostituire `get_g1_supine_robot_cfg()` con `get_g1_supine_robot_cfg_with_delay()` nella riga di assegnazione dell'entita, e impostare `_DR_ACTION_DELAY_ENABLE = True`. Se si usa l'attuatore delayed senza l'evento DR, tutti gli env avranno un lag fisso. Se si usa l'evento DR senza l'attuatore delayed, l'evento non trovera il buffer su cui scrivere. Un commento TODO nel codice segnala esplicitamente questo vincolo.

Disabilitato di default (`_DR_ACTION_DELAY_ENABLE = False`).

---

## Struttura reward completa (19 termini, post-P1)

| Termine | Peso | Tipo |
|---------|------|------|
| base_height_exp | +2.0 | Dense, pelvis height verso 0.728 m |
| body_up_exp | +2.0 | Dense, orientamento eretto |
| stable_success_hold | +3.0 | Sparse, stare in piedi 0.5 s |
| torso_height_exp | +1.0 | Dense, torso_link height verso 1.1 m |
| stand_on_feet | +1.0 | Dense, contatto piedi e altezza < 0.1 m |
| prone_supine_righting (P1.1) | +0.5 | Dense, orientamento gated stage 0 |
| height_progress | +0.5 | Dense shaping, delta h per step |
| is_terminated | -200.0 | Penalita terminazione anticipata |
| joint_pos_limits | -10.0 | Penalita violazione limiti giunti |
| action_saturation | -1.0 | Penalita target fuori limiti soft |
| feet_distance | -0.5 | Stile, distanza tra piedi fuori [0.10, 0.50] m |
| dof_error_when_standing | -0.1 | Stile, deviazione da default gated stage 3 |
| feet_slip | -0.2 | Stile, scivolamento orizzontale piedi |
| action_rate_l2 | -0.05 | Regolarizzazione |
| action_acc_l2 | -0.005 | Regolarizzazione |
| joint_vel_l2 | -0.001 | Regolarizzazione |
| joint_acc_l2 | -2.5e-7 | Regolarizzazione |
| joint_torques_l2 | -2e-5 | Regolarizzazione |

Il principio di gating e il piu importante del reward design: `prone_supine_righting` e attivo solo in stage 0/1 (h < 0.40 m), `dof_error_when_standing` e attivo solo in stage 2/3 (h > 0.60 m). Questo evita che reward pertinenti a fasi diverse del movimento si contraddicano durante la stessa finestra temporale di un episodio.

---

## Domain randomization: stato attuale

| Feature | Stato | Parametri |
|---------|-------|-----------|
| encoder_bias (offset encoder giunto) | Attivo (startup) | bias in [-0.015, 0.015] rad |
| body_com_offset (offset CoM torso) | Attivo (startup) | +/-0.05 m per asse |
| foot_friction | Attivo (startup) | range [0.3, 1.6] |
| motor_strength (P1.4) | Disabilitato | scale [0.90, 1.10] |
| pd_gains (P1.5) | Disabilitato | scale [0.85, 1.15] |
| action_delay (P1.6) | Disabilitato | lag 0–3 steps (0–6 ms) |

Le DR di P1.4/P1.5/P1.6 sono progettate per essere attivate progressivamente dopo che la policy baseline e stabile, nell'ordine: motor_strength (impatto piu contenuto), poi pd_gains, poi action_delay (piu impattante sul comportamento della policy).

---

## File modificati / creati

| File | Tipo | Contenuto |
|------|------|-----------|
| `src/tasks/getup/mdp/rewards.py` | Modificato | Aggiunta `prone_supine_righting` (P1.1) |
| `src/tasks/getup/mdp/metrics.py` | Nuovo/Esteso | 23 funzioni metriche P1.2/P1.3 + 2 helper contact + 15 MetricsTermCfg per contact logging |
| `src/tasks/getup/getup_env_cfg.py` | Modificato | Wiring dei MetricsTermCfg in `make_getup_env_cfg()` |
| `src/tasks/getup/config/g1/env_cfgs.py` | Modificato | 8 nuovi ContactSensorCfg, nconmax/contact_sensor_maxmatch aumentati, flag DR P1.4/P1.5/P1.6 |
| `src/assets/robots/unitree_g1/g1_constants.py` | Modificato | `G1_ARTICULATION_DELAYED`, `_wrap_with_delay`, `get_g1_supine_robot_cfg_with_delay` |

---

## Architettura di simulazione (riferimento)

- Timestep MuJoCo: 0.002 s, decimation 5 → controllo a 100 Hz
- Episodi: 10 s (1000 step di controllo)
- Reset: campiona casualmente SUPINE_KEYFRAME o PRONE_KEYFRAME con perturbazione di giunti, root e velocita iniziale; yaw completamente casuale
- Collisioni: `FULL_COLLISION` con `condim=3` + friction elevata su mani/polsi/gomiti/pelvis/torso (supporto al rialzarsi), `condim=1` su tutti gli altri geom
- Osservazioni actor (500-dim, history x4): IMU ang_vel, projected_gravity, body_height, joint_pos, joint_vel, last_action, feet_contact, pd_tracking_error
- Azioni: `LowPassJointPositionAction` (residuo su default, EMA alpha=0.5), scale per-giunto da `0.25 * effort_limit / stiffness`
- Terminazioni: timeout, NaN, joint_vel_explosion (>50 rad/s), ground_penetration (<-0.3 m), standing_fall_timeout (caduta dopo standing per 30 step)

---

## Note

**Coerenza soglie reward/metrics.** Le soglie di stage in `metrics.py` (`_H_SUPPORT=0.20`, `_H_RISING=0.40`, `_H_STANDING=0.65`) devono corrispondere a quelle usate nei reward. Un disallineamento comporterebbe un mislabelling dei stage nei log. Un commento nel file `metrics.py` avverte esplicitamente di questo vincolo.

**Torque saturation con DR attiva.** Quando P1.4 sara attivato, `actuator_forcerange` sara per-env e avra shape `[n_envs, n_actuators, 2]`. La funzione `_get_effort_limits` gestisce gia entrambe le forme, ma usa l'env 0 come riferimento. Per metriche per-env accurate con DR attiva si potrebbe in futuro usare il limite dell'env corrispondente invece di env 0.

**Contact logging e reward futuri.** Il contact logging e attualmente puramente diagnostico (nessun reward usa i sensori di contatto delle parti del corpo). In futuro e possibile derivare reward differenziati da questi sensori, per esempio penalizzare il contatto prolungato della testa o incentivare l'uso delle mani per spingere. L'infrastruttura sensor e gia in place; aggiungere un reward richiederebbe solo una nuova funzione in `rewards.py` che legge `env.scene["contact_head"].data.found`.

**nconmax e dimensionamento.** Il valore 100 e un upper bound conservativo per 4–8 env in debug. In training con 4096 env su GPU, `nconmax` e tipicamente un parametro per-env nel backend MuJoCo (ogni env ha il suo solver isolato), quindi il valore non scala linearmente con il numero di env.
