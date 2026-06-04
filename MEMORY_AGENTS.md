Da creazione di SUPINE KEYFRAME e mdp_rewards

## Cosa è stato fatto

Porting completo del task "get-up" (alzarsi da terra) da Isaac Gym / legged_gym (repository HumanUP originale) a MuJoCo / mjlab. Il risultato è un task registrato come `Unitree-G1-GetUp` che fa iniziare il robot G1 in posizione supina e lo addestra a rialzarsi in piedi utilizzando PPO.

---

## Perché è stato fatto

Il progetto HumanUP contiene già una policy "get-up" funzionante addestrata con Isaac Gym (`simulation/legged_gym/legged_gym/envs/g1waist/g1waist_up.py`). Porting del task su mjlab permette di:

- Addestrare la stessa skill su MuJoCo, che è il simulatore di riferimento per il deployment su hardware reale in questo setup.
- Riutilizzare l'infrastruttura esistente di mjlab (runner, esportazione ONNX, domain randomization) senza dover mantenere due codebase di training.
- Avere reward e comportamento numericamente coerenti con quelli usati in Isaac Gym, rendendo più facile comparare le due policy e fare fine-tuning.

---

## Dettagli tecnici

### Posa supina iniziale (SUPINE_KEYFRAME)

La posa di partenza è l'aspetto più delicato del porting perché la rappresentazione dei quaternioni differisce tra i due framework:

- **Isaac Gym** usa la convenzione `(x, y, z, w)`: il valore di `root_states` rilevato durante il training HumanUP era `(-0.00982, 0.49986, 0.01753, -0.86587)`.
- **MuJoCo** usa la convenzione `(w, x, y, z)`: il quaternione è stato riordinato in `(-0.866, -0.010, 0.500, 0.018)`.

Questo quaternione corrisponde a una rotazione di circa 120° attorno all'asse Y locale, che posiziona il robot con la schiena a terra e il viso rivolto verso l'alto — cioè la postura supina coerente con il training originale.

L'altezza iniziale è `z=0.1` m anziché 0: questo margine è necessario perché MuJoCo richiede che il corpo non compenetri il terreno al momento della creazione della scena; una piccola elevazione permette al solver di contatto di risolvere la penetrazione iniziale senza instabilità numeriche.

Il campo `joint_pos={".*": 0.0}` azzera tutti e 29 i gradi di libertà, rispecchiando il comportamento del reset di Isaac Gym che parte dalla posa di default.

### Struttura del task

La struttura di cartelle rispecchia il pattern già stabilito dal task `velocity`:

```
src/tasks/getup/
├── getup_env_cfg.py       factory function — configurazione base robot-agnostica
├── mdp/                   reward, observations, terminations custom
├── rl/                    runner con export ONNX automatico al salvataggio
└── config/g1/             override G1-specifici + registrazione task
```

Il pattern a due livelli (base generico + override per robot) è la stessa architettura usata dal task velocity. La factory `make_getup_env_cfg()` produce una configurazione robot-agnostica; `unitree_g1_getup_env_cfg()` applica poi tutto ciò che è specifico del G1 (nomi dei body, scale degli attuatori, sensori di contatto, domain randomization).

### Traduzione API Isaac Gym → mjlab

Ogni reward è stato tradotto mantenendo la stessa logica matematica; solo le chiamate API sono cambiate:

| Grandezza fisica | Isaac Gym | mjlab |
|---|---|---|
| Altezza pelvis | `root_states[:, 2]` | `asset.data.root_link_pos_w[:, 2]` |
| Altezza corpo arbitrario | `rigid_body_states[:, idx, 2]` | `asset.data.body_link_pos_w[:, body_ids, 2]` |
| Gravità proiettata nel frame corpo | `projected_gravity[:, 2]` | `asset.data.projected_gravity_b[:, 2]` |
| Contatto piede | `contact_forces > threshold` | `ContactSensor.data.found > 0` |
| Posizione sito (punto anatomico) | `rigid_body_states[:, foot_idx, :]` | `asset.data.site_pos_w[:, site_ids, :]` |
| Posizione giunti | `dof_pos` | `asset.data.joint_pos` |
| Posizione default giunti | `default_dof_pos` | `asset.data.default_joint_pos` |

### Reward shaping

Cinque reward custom sono stati portati da HumanUP:

1. **`base_height_exp`** (peso 2.0): incentiva il pelvis ad alzarsi verso 0.728 m (altezza in piedi G1). Formula `exp(clamp(z, 0, target)) - 1`, che dà segnale zero a terra e cresce monotonamente.
2. **`torso_height_exp`** (peso 1.0): stesso schema ma applicato a `torso_link`, il corpo più alto del G1 tracciato nel modello. Target 1.1 m.
3. **`body_up_exp`** (peso 2.0): `exp(-projected_gravity_b_z)` — massimo quando il vettore gravità nel frame corpo punta verso il basso (robot eretto), minimo quando punta verso l'alto (robot rovesciato).
4. **`stand_on_feet`** (peso 1.0): reward binario che richiede *sia* il contatto di entrambi i piedi con il terreno *sia* che l'altezza dei siti piede sia sotto 0.1 m. Evita che la policy impari a stare "in piedi" sulle mani.
5. **`dof_error_when_standing`** (peso -0.1): penalità sulla deviazione dalla posa default, attivata **solo** quando il pelvis supera 0.6 m. Questo è il punto critico del porting: penalizzare le pose strane mentre il robot si sta ancora alzando contraddirebbe il reward di altezza, quindi la penalità è condizionale alla postura già raggiunta.

Reward mjlab built-in aggiunti: `action_rate_l2`, `joint_torques_l2`, `is_terminated`, `joint_pos_limits`.

### Differenze architetturali rispetto al task velocity

Il get-up è strutturalmente più semplice del velocity perché manca di tutto ciò che riguarda la locomozione direzionale:

- **Nessun comando**: `commands={}` — l'obiettivo è fisso ("alzati"), non parametrico.
- **Nessuna terminazione per orientazione sbagliata**: `bad_orientation` sparerebbe al primo step perché il robot inizia a terra. Solo `time_out` e `nan_detection`.
- **Terreno piano**: nessun rough terrain, nessun raycast sensor per la stima del terreno, nessun height scan nell'osservazione.
- **Timestep più fine**: 0.002 s (vs 0.005 s del velocity), con decimation=5 → controllo a 100 Hz. La scelta è motivata dalla natura del task: alzarsi da terra richiede transizioni di contatto rapide che beneficiano di un timestep di integrazione più piccolo.
- **Episodio più corto**: 10 s invece di 20 s — sufficiente per alzarsi, ma riduce il tempo di campionamento sprecato su episodi dove il robot non impara nulla.
- **CCD abilitato**: `ccd_iterations=500` e `nconmax=48` per gestire le collisioni durante le fasi di rotolamento sul terreno, molto più frequenti che nel walking.

### Osservazione personalizzata

Aggiunta l'osservazione `body_height` (altezza pelvis, scalare per ogni env) che restituisce `root_link_pos_w[:, 2:3]` con rumore uniforme ±0.02 m. Questa osservazione non è presente nel task velocity ma è essenziale qui: la policy deve sapere a che quota si trova il pelvis per imparare a coordinare la sequenza di movimenti necessaria ad alzarsi.

### Runner e salvataggio

`GetUpOnPolicyRunner` estende `MjlabOnPolicyRunner` solo per l'override del metodo `save()`: ad ogni checkpoint viene esportata automaticamente una policy ONNX con i metadati del run (nome del run wandb incluso). Lo stesso pattern è usato dal runner del task velocity.

---

## File modificati / creati

| File | Tipo | Descrizione |
|---|---|---|
| `src/assets/robots/unitree_g1/g1_constants.py` | Modificato | Aggiunto `SUPINE_KEYFRAME` e `get_g1_supine_robot_cfg()` |
| `src/assets/robots/__init__.py` | Modificato | Esporta `get_g1_supine_robot_cfg` |
| `src/tasks/getup/__init__.py` | Creato | Package marker |
| `src/tasks/getup/getup_env_cfg.py` | Creato | Factory base robot-agnostica |
| `src/tasks/getup/mdp/__init__.py` | Creato | Re-export mjlab.envs.mdp + custom |
| `src/tasks/getup/mdp/rewards.py` | Creato | 5 reward functions portate da HumanUP |
| `src/tasks/getup/mdp/observations.py` | Creato | Osservazione `body_height` |
| `src/tasks/getup/mdp/terminations.py` | Creato | Placeholder (solo built-in usati) |
| `src/tasks/getup/rl/__init__.py` | Creato | Package marker |
| `src/tasks/getup/rl/runner.py` | Creato | `GetUpOnPolicyRunner` con export ONNX |
| `src/tasks/getup/config/__init__.py` | Creato | Package marker |
| `src/tasks/getup/config/g1/__init__.py` | Creato | Registra `Unitree-G1-GetUp` |
| `src/tasks/getup/config/g1/env_cfgs.py` | Creato | Override G1-specifici |
| `src/tasks/getup/config/g1/rl_cfg.py` | Creato | Config PPO (512-256-128, lr=1e-3, adaptive) |

---

## Note e considerazioni

**Conversione quaternione**: la conversione da Isaac Gym `(x,y,z,w)` a MuJoCo `(w,x,y,z)` è banale ma critica. Un errore qui produce un robot che parte con l'orientazione sbagliata e non impara nulla. Il valore `(-0.866, -0.010, 0.500, 0.018)` rappresenta una rotazione circa 120° sull'asse Y: il componente W=-0.866 indica una rotazione di ~150° (poiché `cos(θ/2)=W`), che piega il robot verso l'indietro fino a farlo giacere con la schiena a terra.

**Domain randomization al reset**: `reset_base` usa `yaw` random su tutto il range `[-π, π]`. Questo è intenzionale: la policy deve imparare ad alzarsi indipendentemente dall'orientamento planare iniziale, altrimenti fallisce in deploy quando il robot cade in direzioni arbitrarie.

**Friction randomization**: il range `(0.3, 1.6)` con `shared_random=True` applica la stessa perturbazione a tutti i geom del piede contemporaneamente. Questo evita asimmetrie artificiali tra piede sinistro e destro durante il training.

**Pesi reward**: i pesi `base_height_exp` e `body_up_exp` sono entrambi 2.0, doppi rispetto a `torso_height_exp` e `stand_on_feet`. Questa gerarchia mette l'orientazione e l'altezza del pelvis come segnali primari, con la posizione del torso e il contatto dei piedi come rinforzo secondario — coerente con la strategia HumanUP originale.

**Compatibilità ONNX**: l'export automatico al checkpoint è necessario per il deployment diretto su hardware G1 senza passi manuali aggiuntivi.


