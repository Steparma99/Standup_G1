Piano di Integrazione HoST → HumanUP G1 29 DOF

   Contesto

   Il progetto HumanUP usa unitree_rl_mjlab (IsaacLab + backend MuJoCo) con rsl_rl PPO per addestrare un G1 a 29 DOF ad alzarsi da terra. HoST è una repository parallela che usa IsaacGym con un G1 a 23 DOF — framework completamente diverso, quindi il codice non è portabile 1:1. Quello che si porta sono i concetti e le strutture dati, tradotti nelle API di IsaacLab/MuJoCo.                                                                                                       ↑

   I 6 DOF extra nel G1 29 rispetto al 23 sono: +2 waist (roll/pitch, HoST ha solo waist_yaw) + +1 wrist per braccio (wrist_pitch + wrist_yaw vs solo wrist_roll in HoST) = 6 DOF in più.
                                                                                                                          ↑
   Priorità di implementazione: ordinate per valore/rischio — le prime non dipendono dalle ultime.

   ---                                                                                                                    ↑
   Nota Architetturale Critica

   HoST usa IsaacGym (NVIDIA, solo GPU), HumanUP usa IsaacLab con backend MuJoCo.                                         ↑
   - Le classi di configurazione (EventTermCfg, RewardTermCfg, ObsTermCfg) sono IsaacLab
   - I sensori di contatto sono gestiti via ContactSensorCfg
   - Le forze esterne si applicano via scene.robot.apply_external_force_and_torque()                                      ↑
   - I siti anatomici in MuJoCo XML si definiscono con <site> (massless markers)
                                                                                                                          ↑
   ---
   Feature 1 — Style & Symmetry Rewards
                                                                                                                          ↑
   Priorità: ALTA | Rischio: BASSO | Dipendenze: nessuna

   Perché                                                                                                                 ↑

   Rewards di stile migliorano la qualità del movimento durante il sim2real: penalizzano posture bizzarre che funzionano in simulazione ma collassano sul robot reale. HoST li usa come gruppo separato con peso 1.0 moltiplicativo.               ↑

   Concetti da HoST
                                                                                                                          ↑
   - waist_deviation: penalità se waist roll/pitch/yaw escono da soglia (es. >0.8 rad)
   - hip_symmetry: |hip_pitch_L - hip_pitch_R| — simmetria sinistra/destra
   - shoulder_deviation: braccia vicine al corpo quando in piedi (shoulder_roll ~ 0)                                      ↑
   - shank_orientation: tibia verticale durante l'alzata (dot product shank frame con z_world)
   - feet_level: ankle_roll vicino a 0 quando in piedi                                                                    ↑
   - feet_under_body: piedi proiettati sotto il pelvis (distanza XY piccola)
   - feet_distance: piedi non troppo distanti tra loro                                                                    ↑

   Implementazione                                                                                                        ↑

   File: unitree_rl_mjlab/src/tasks/getup/mdp/rewards.py
   Aggiungere le seguenti funzioni reward (tutte stateless, pure math su env.scene):

   def waist_deviation(env, waist_pitch_weight, waist_roll_weight, threshold):                                            ↑
       # joint_pos per waist_pitch_joint e waist_roll_joint
       # penalità esponenziale se |pos| > threshold
                                                                                                                          ↑
   def body_symmetry(env, weight):
       # |hip_pitch_L - hip_pitch_R| + |knee_L - knee_R| + |ankle_pitch_L - ankle_pitch_R|                                ↑

   def shank_orientation(env, weight):                                                                                    ↑
       # vettore knee→ankle in world frame, dot con [0,0,1]
       # reward se tibia è vicina alla verticale                                                                          ↑

   def feet_under_pelvis(env, weight):
       # pos_pelvis_xy - mean(pos_feet_xy), penalità se distanza > threshold                                              ↑

   def shoulder_symmetry(env, weight):
       # |shoulder_roll_L - shoulder_roll_R| penalità                                                                     ↑

   File: unitree_rl_mjlab/src/tasks/getup/getup_env_cfg.py
   Aggiungere alla sezione rewards:                                                                                       ↑
   # Style group (aggiungere con pesi bassi inizialmente: 0.1–0.5)
   waist_deviation = RewardTermCfg(func=rewards.waist_deviation, weight=-0.3, ...)                                        ↑
   body_symmetry = RewardTermCfg(func=rewards.body_symmetry, weight=-0.2, ...)
   shank_orientation = RewardTermCfg(func=rewards.shank_orientation, weight=0.5, ...)                                     ↑
   feet_under_pelvis = RewardTermCfg(func=rewards.feet_under_pelvis, weight=-0.3, ...)

   File: unitree_rl_mjlab/src/tasks/getup/mdp/metrics.py
   Aggiungere metriche style/* per TensorBoard (pattern identico alle metriche esistenti).
                                                                                                                          ↑
   Indici DOF utili (da g1_constants.py)

   - waist_yaw_joint, waist_roll_joint, waist_pitch_joint → indici 12–14                                                  ↑
   - left_hip_pitch_joint vs right_hip_pitch_joint
   - left_ankle_roll_joint vs right_ankle_roll_joint
                                                                                                                          ↑
   ---
   Feature 2 — Extended Init Positions
                                                                                                                          ↑
   Priorità: ALTA | Rischio: BASSO | Dipendenze: nessuna

   Perché                                                                                                                 ↑

   Attualmente il robot parte solo SUPINE o PRONE. Aggiungere SIDE_LEFT, SIDE_RIGHT (sdraiato su un fianco) rende la policy più robusta e copre scenari reali. HoST prevede anche SEATED e CROUCHED come keyframe.                                 ↑

   Keyframe da aggiungere
                                                                                                                          ↑
   I valori esatti di quaternion/joint vanno determinati empiricamente (o da MuJoCo keyframe editor), ma le stime ragionevoli sono:
                                                                                                                          ↑
   ┌────────────┬───────────┬─────────────────────────┬─────────────────────────┐
   │  Keyframe  │ pos Z (m) │  Quaternion (w,x,y,z)   │          Note           │
   ├────────────┼───────────┼─────────────────────────┼─────────────────────────┤                                         ↑
   │ SIDE_LEFT  │ ~0.12     │ (0.7071, -0.7071, 0, 0) │ Fianco sinistro a terra │
   ├────────────┼───────────┼─────────────────────────┼─────────────────────────┤
   │ SIDE_RIGHT │ ~0.12     │ (0.7071, 0.7071, 0, 0)  │ Fianco destro a terra   │                                         ↑
   ├────────────┼───────────┼─────────────────────────┼─────────────────────────┤
   │ SEATED     │ ~0.45     │ (1, 0, 0, 0)            │ Seduto con gambe tese   │
   └────────────┴───────────┴─────────────────────────┴─────────────────────────┘                                         ↑

   File: unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py
   Aggiungere alla lista KEYFRAMES:                                                                                       ↑
   KEYFRAME_SIDE_LEFT = {
       "qpos": [...],  # joint angles con hip_roll_L ~1.2, hip_roll_R ~-0.3                                               ↑
       "quat": [0.7071, -0.7071, 0.0, 0.0],
       "pos": [0.0, 0.0, 0.12],                                                                                           ↑
   }
   KEYFRAME_SIDE_RIGHT = { ... }                                                                                          ↑

   File: unitree_rl_mjlab/src/tasks/getup/mdp/events.py
   Nella funzione reset_to_random_keyframe, estendere il campionamento:
   # Attuale: campiona random tra SUPINE e PRONE (0.5/0.5)                                                                ↑
   # Nuovo: campiona tra [SUPINE, PRONE, SIDE_LEFT, SIDE_RIGHT] con probabilità configurabile
   keyframe_probs = env.cfg.init_pose_probs  # es. [0.3, 0.3, 0.2, 0.2]
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/config/g1/env_cfgs.py
   Aggiungere parametro init_pose_probs configurabile e flag _ENABLE_SIDE_POSES.                                          ↑

   Come trovare i valori giusti

   1. Aprire il viewer MuJoCo: python scripts/play.py Unitree-G1-GetUp --viewer native
   2. Muovere il robot a mano con il viewer                                                                               ↑
   3. Leggere qpos dalla console con print(env.scene.robot.data.joint_pos)

   ---                                                                                                                    ↑
   Feature 3 — Anatomical Keypoints (Siti Anatomici)
                                                                                                                          ↑
   Priorità: MEDIA | Rischio: MEDIO | Dipendenze: nessuna, ma abilita Feature 1 avanzata

   Perché

   HoST usa 17 "keyframe bodies" (link zero-massa) per tracciare pose anatomiche e per futura motion imitation. In MuJoCo equivale a sites — marker massless attaccati a body links. Servono per:
   1. Reward di pose più precisi (distanza tra keypoint predetto e target)
   2. Osservazioni più ricche per il critic                                                                               ↑
   3. Base per futura motion imitation (HoST approach avanzato)
                                                                                                                          ↑
   I 17 siti di HoST (adattati per G1 29 DOF)
                                                                                                                          ↑
   head, torso, pelvis
   collar_L, collar_R
   shoulder_L, shoulder_R
   elbow_L, elbow_R                                                                                                       ↑
   wrist_L, wrist_R
   hip_L, hip_R
   knee_L, knee_R                                                                                                         ↑
   ankle_L, ankle_R

   Implementazione MuJoCo                                                                                                 ↑

   File: unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml
   Per ogni link rilevante, aggiungere un <site> con offset ergonomico:                                                   ↑
   <!-- Nel body "head_link" -->
   <site name="kp_head" pos="0 0 0.05" size="0.01"/>
                                                                                                                          ↑
   <!-- Nel body "pelvis" -->
   <site name="kp_pelvis" pos="0 0 0" size="0.01"/>
                                                                                                                          ↑
   <!-- Nel body "left_knee_link" -->
   <site name="kp_knee_l" pos="0 0 -0.05" size="0.01"/>
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/assets/robots/unitree_g1/g1_constants.py
   KEYPOINT_SITE_NAMES = [
       "kp_head", "kp_torso", "kp_pelvis",                                                                                ↑
       "kp_collar_l", "kp_collar_r",
       "kp_shoulder_l", "kp_shoulder_r",
       "kp_elbow_l", "kp_elbow_r",                                                                                        ↑
       "kp_wrist_l", "kp_wrist_r",
       "kp_hip_l", "kp_hip_r",
       "kp_knee_l", "kp_knee_r",                                                                                          ↑
       "kp_ankle_l", "kp_ankle_r",
   ]
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/getup_env_cfg.py
   Aggiungere un FrameTransformerCfg o accedere ai body positions direttamente:
   # IsaacLab: accesso a body positions tramite articulation data                                                         ↑
   # env.scene.robot.data.body_pos_w[:, body_idx, :]
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/mdp/observations.py
   Aggiungere funzione opzionale per critic privilegiato:                                                                 ↑
   def keypoint_positions(env):
       # Ritorna posizioni world-frame di tutti i siti anatomici                                                          ↑
       # Shape: (num_envs, 17*3) = (N, 51)
       body_indices = env.cfg.keypoint_body_indices
       return env.scene.robot.data.body_pos_w[:, body_indices, :].reshape(N, -1)
                                                                                                                          ↑
   ---
   Feature 4 — Assistance Curriculum (Forza di Supporto)
                                                                                                                          ↑
   Priorità: ALTA | Rischio: MEDIO | Dipendenze: nessuna
                                                                                                                          ↑
   Perché
                                                                                                                          ↑
   Questo è il contributo più importante di HoST per la convergenza del training. Il robot G1 pesa ~35 kg. Alzarsi da terra è un task difficile — il policy gradient è molto sparso all'inizio perché il robot raramente riesce ad alzarsi per caso. Applicare una forza verticale sul torso (simula un "aiuto" fisico che si ritira gradualmente) risolve il problema di esplorazione.                                                                                                          ↑

   Meccanismo HoST
                                                                                                                          ↑
   Forza iniziale: 100 N (~ 30% peso robot)
   Applicata a: torso_link, direzione +Z (verso l'alto)                                                                   ↑
   Condizione: solo dopo i primi 30 step (lasciar cadere liberamente),
               e solo se orientazione ok (gravity_z < -0.8) — skip per prone                                              ↑
   Decay: -20 N per ogni reset dove mean(head_height) > 0.9 m
   Floor: 0 N (si azzera completamente)

   Implementazione IsaacLab/MuJoCo
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/getup_env_cfg.py
   Aggiungere sezione curriculum:
   @dataclass                                                                                                             ↑
   class AssistanceCurriculumCfg:
       enabled: bool = True                                                                                               ↑
       initial_force_n: float = 80.0      # N, più basso che HoST (G1 pesa ~35kg)
       force_decay_per_success: float = 5.0   # N per reset riuscito                                                      ↑
       force_min: float = 0.0
       success_height_threshold: float = 0.85  # m, head height per "riuscito"                                            ↑
       unactuated_steps: int = 20        # step passivi iniziali prima di applicare
       apply_to_body: str = "torso_link"

   File: unitree_rl_mjlab/src/tasks/getup/mdp/events.py
   def apply_assistance_force(env):                                                                                       ↑
       """Chiamato ogni step, applica forza verticale condizionale."""
       if not env.cfg.assistance_curriculum.enabled:
           return                                                                                                         ↑
       cfg = env.cfg.assistance_curriculum
                                                                                                                          ↑
       # Skip unactuated phase
       if env.episode_length_buf < cfg.unactuated_steps:                                                                  ↑
           return

       # Condizione: solo se non ancora in piedi (height check)
       # Applica forza solo agli env che non hanno ancora superato il threshold                                           ↑

       force = env.assistance_force  # (num_envs,) tensor, inizializzato in reset
       torso_idx = env.torso_body_idx                                                                                     ↑

       forces = torch.zeros(env.num_envs, env.scene.robot.num_bodies, 3, device=env.device)                               ↑
       forces[:, torso_idx, 2] = force  # +Z upward
       env.scene.robot.apply_external_force_and_torque(forces, torch.zeros_like(forces))                                  ↑

   def decay_assistance_on_reset(env, env_ids):
       """Chiamato al reset: se policy ha avuto successo, riduci forza."""
       success_mask = env.metrics["ever_stood"][env_ids]  # o head_height > threshold                                     ↑
       env.assistance_force[env_ids[success_mask]] -= env.cfg.assistance_curriculum.force_decay_per_success
       env.assistance_force[env_ids].clamp_(min=0.0)
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/mdp/observations.py
   Aggiungere al critic privilegiato: assistance_force_normalized (forza attuale / forza iniziale, scalare per env).      ↑

   File: unitree_rl_mjlab/src/tasks/getup/mdp/metrics.py                                                                  ↑
   Aggiungere metrica curriculum/assistance_force_mean per vedere il decay su TensorBoard.

   Nota sull'API IsaacLab

   Il metodo esatto è ArticulationData.apply_external_force_and_torque() — verificare la versione di IsaacLab usata nel progetto (pip show isaaclab o vedere requirements.txt). L'alternativa è scrivere nell'articulation external_force_b direttamente se disponibile.
                                                                                                                          ↑
   ---
   Feature 5 — Staged Reward Restructuring (Gruppi Reward)                                                                ↑

   Priorità: MEDIA | Rischio: MEDIO-ALTO | Dipendenze: Feature 1 (style rewards)

   Perché
                                                                                                                          ↑
   HoST organizza i reward in 4 gruppi con pesi separati: task × regu × style × post_task. Il reward task è moltiplicativo — se il robot non è orientato/alto, tutto il resto vale zero. Questo forza una gerarchia di apprendimento chiara.
                                                                                                                          ↑
   HumanUP attuale ha già staging (gating su stage0-3) ma in forma additiva. La ristrutturazione rende il curriculum più esplicito e manutenibile.                                                                                              ↑

   Struttura Target                                                                                                       ↑

   reward_groups = {                                                                                                      ↑
       "task":      weight=2.5,  # height + orientation (moltiplicativo sugli altri)
       "regu":      weight=0.1,  # smoothness, torque, velocity (additivo)
       "style":     weight=1.0,  # waist, symmetry, feet (additivo, gated post-stage1)
       "post_task": weight=1.0,  # stable hold, com over support (additivo, gated stage3+)
   }                                                                                                                      ↑

   Implementazione
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/getup_env_cfg.py
   Aggiungere reward_group tag ai RewardTermCfg esistenti:
   # Già esistente, aggiungere params={"group": "task"}                                                                   ↑
   base_height_exp = RewardTermCfg(..., params={"group": "task"})
   body_up_exp = RewardTermCfg(..., params={"group": "task"})
   # etc.                                                                                                                 ↑

   File: unitree_rl_mjlab/src/tasks/getup/mdp/rewards.py
   Aggiungere funzione aggregatrice:                                                                                      ↑
   def compute_grouped_reward(env):
       """Calcola reward totale con struttura a gruppi."""
       task_r = sum(task_group_rewards)                                                                                   ↑
       regu_r = sum(regu_group_rewards)
       style_r = sum(style_group_rewards)                                                                                 ↑
       post_r  = sum(post_task_group_rewards) * (stage >= 3)
       return task_r * (cfg.regu_w * regu_r + cfg.style_w * style_r + cfg.post_w * post_r)
                                                                                                                          ↑
   Attenzione: Questa è una modifica profonda al loop di training. Da fare DOPO che un primo run NVIDIA ha già convergito bene con la struttura attuale additiva.
                                                                                                                          ↑
   ---
   Feature 6 — Extended Domain Randomization
                                                                                                                          ↑
   Priorità: MEDIA | Rischio: BASSO (è già architetturato) | Dipendenze: run stabile

   Perché                                                                                                                 ↑

   I flag DR esistenti in env_cfgs.py sono tutti _ENABLE = False. Oltre ad attivarli, HoST aggiunge:
   - Randomizzazione massa links (±20%)                                                                                   ↑
   - Payload mass (oggetto portato, ±2-5 kg)
   - Spostamento CoM (-3/+3 cm)                                                                                           ↑
   - Attrito suolo (0.1-1.0)
   - Push random durante episodio                                                                                         ↑

   Implementazione

   File: unitree_rl_mjlab/src/tasks/getup/config/g1/env_cfgs.py
   I flag attuali da attivare (in ordine, uno per run):                                                                   ↑
   1. _DR_MOTOR_STRENGTH_ENABLE = True (±10% torque)
   2. _DR_PD_GAINS_ENABLE = True (±15% Kp, Kd)
   3. _DR_ACTION_DELAY_ENABLE = True (delay 0-2 step)                                                                     ↑

   Nuovi da aggiungere:                                                                                                   ↑
   _DR_MASS_ENABLE = False
   _DR_FRICTION_ENABLE = False
   _DR_PUSH_ENABLE = False

   # Config valori:                                                                                                       ↑
   DR_LINK_MASS_RANGE = [0.8, 1.2]      # ±20% per link
   DR_FRICTION_RANGE = [0.4, 1.2]       # range coefficiente attrito
   DR_PUSH_INTERVAL = 200               # step tra push                                                                   ↑
   DR_PUSH_VEL_XY = [0.0, 0.8]         # m/s impulso laterale
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/mdp/events.py
   Aggiungere funzioni randomize_physics_mass(), randomize_friction(), push_robot_velocity() seguendo il pattern delle funzioni randomize esistenti.

   ---
   Feature 7 — Task Family Structure (Multi-Task)                                                                         ↑

   Priorità: BASSA | Rischio: BASSO | Dipendenze: Feature 2
                                                                                                                          ↑
   Perché
                                                                                                                          ↑
   HoST ha task separati: GetUp_Ground, GetUp_Prone, GetUp_Platform, GetUp_Slope, GetUp_Wall. Ogni task ha la stessa struttura MDP ma config diversa (init pose, terrain, reward weights, forza assistenza). Questa separazione permette curriculum progressivo.

   Struttura File                                                                                                         ↑

   unitree_rl_mjlab/src/tasks/getup/
   ├── config/g1/                                                                                                         ↑
   │   ├── env_cfgs.py          # ← attuale (GetUp_Ground, supine+prone)
   │   ├── env_cfgs_prone.py    # ← solo prone, assistenza maggiore
   │   ├── env_cfgs_side.py     # ← solo lato, task più difficile                                                         ↑
   │   └── env_cfgs_wall.py     # ← con wall terrain, init crouched
                                                                                                                          ↑
   File: unitree_rl_mjlab/src/tasks/getup/config/g1/env_cfgs_prone.py
   class G1GetUpEnvCfgProne(G1GetUpEnvCfg):
       init_pose_probs = [0.0, 1.0, 0.0, 0.0]  # solo PRONE
       assistance_curriculum: AssistanceCurriculumCfg = AssistanceCurriculumCfg(
           initial_force_n=100.0,  # più alta per prone (più difficile)                                                   ↑
           no_orientation_check=True,  # come HoST prone config
       )                                                                                                                  ↑

   File: unitree_rl_mjlab/unitree_rl_mjlab/tasks/__init__.py (o equivalente)                                              ↑
   Registrare task ID aggiuntivi:
   "Unitree-G1-GetUp-Prone", "Unitree-G1-GetUp-Side", "Unitree-G1-GetUp-Wall"                                             ↑

   ---
   Feature 8 — Terrain Variants

   Priorità: BASSA | Rischio: ALTO | Dipendenze: Feature 7                                                                ↑

   Perché
                                                                                                                          ↑
   IsaacLab supporta generazione procedurale di terreni via TerrainImporterCfg. HoST ha slope, platform e wall come mesh trimesh. In MuJoCo i terreni si definiscono come heighfield o mesh inclusi nella scene XML.                            ↑

   Implementazione

   Step 1 — Slope:
   Generare un piano inclinato (5-15°) come <geom type="plane" ...> nel scene XML con rotazione, o usare HeightFieldTerrainCfg se disponibile in IsaacLab con backend MuJoCo.

   Step 2 — Platform:                                                                                                     ↑
   Aggiungere <geom type="box"> al di sopra del piano base nel scene XML per piattaforme elevate.

   Step 3 — Wall:                                                                                                         ↑
   Aggiungere parete verticale <geom type="box" size="0.05 2.0 1.5"> a distanza appropriata.
                                                                                                                          ↑
   File scene da creare:
   unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/                                                                    ↑
   ├── scene_g1.xml          # ← attuale (piano piatto)
   ├── scene_g1_slope.xml    # ← piano inclinato 10°
   ├── scene_g1_platform.xml # ← piattaforma elevata
   └── scene_g1_wall.xml     # ← con parete                                                                               ↑

   Nota: Questa feature ha il rischio maggiore perché richiede modifiche XML e test fisici approfonditi. Da fare solo dopo che tutte le altre feature sono stabili.                                                                               ↑

   ---                                                                                                                    ↑
   Ordine di Implementazione Consigliato
                                                                                                                          ↑
   ┌─────┬─────────────────────────────┬─────────────┬─────────────────┬───────────────────────────────────────┐
   │  #  │           Feature           │ Ore stimate │   Dipendenze    │                 Note                  │
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 1   │ Style Rewards               │ 3-4h        │ —               │ Inizia da qui, massimo valore/rischio │          ↑
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 2   │ Extended Init Positions     │ 2-3h        │ —               │ Trovare quaternion via viewer         │          ↑
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 3   │ Keypoints Anatomici         │ 4-5h        │ —               │ Fondamentale per feature future       │          ↑
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 4   │ Assistance Curriculum       │ 5-6h        │ —               │ Più impattante per convergenza        │          ↑
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 5   │ Extended DR                 │ 2-3h        │ run stabile     │ Attivare uno per volta                │
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 6   │ Task Family Structure       │ 2h          │ #2              │ Solo refactoring organizzativo        │
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤          ↑
   │ 7   │ Staged Reward Restructuring │ 4-5h        │ #1, run stabile │ Rischio più alto, dopo convergenza    │
   ├─────┼─────────────────────────────┼─────────────┼─────────────────┼───────────────────────────────────────┤
   │ 8   │ Terrain Variants            │ 6-8h        │ #6              │ Per ultimo, più complesso             │          ↑
   └─────┴─────────────────────────────┴─────────────┴─────────────────┴───────────────────────────────────────┘
                                                                                                                          ↑
   ---
   File Critici da Modificare

   ┌─────────────────────────────────────────────────┬────────────────────┐
   │                      File                       │      Feature       │                                               ↑
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/tasks/getup/mdp/rewards.py                  │ #1, #5, #7         │
   ├─────────────────────────────────────────────────┼────────────────────┤                                               ↑
   │ src/tasks/getup/mdp/events.py                   │ #2, #4, #6         │
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/tasks/getup/mdp/observations.py             │ #3                 │
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/tasks/getup/mdp/metrics.py                  │ #1, #4             │                                               ↑
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/tasks/getup/getup_env_cfg.py                │ #1, #3, #4, #5, #7 │
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/tasks/getup/config/g1/env_cfgs.py           │ #2, #4, #5, #6     │
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/assets/robots/unitree_g1/g1_constants.py    │ #2, #3             │
   ├─────────────────────────────────────────────────┼────────────────────┤
   │ src/assets/robots/unitree_g1/xmls/g1.xml        │ #3, #8             │
   ├─────────────────────────────────────────────────┼────────────────────┤                                             │ src/tasks/getup/config/g1/env_cfgs_*.py (nuovi) │ #6, #8             │
   └─────────────────────────────────────────────────┴────────────────────┘

   ---
   Verifica (Come testare ogni feature)

   1. Style Rewards: smoke test (10 iter, cpu) — TensorBoard style/* non NaN, valori negativi piccoli
   2. Init Positions: --viewer native — visivamente le 4 pose sembrano corrette fisicamente
   3. Keypoints: aggiungere print(env.scene.robot.data.body_pos_w.shape) per verificare indici
   4. Assistance Curriculum: TensorBoard curriculum/assistance_force_mean deve decrescere nel tempo
   5. DR: smoke test con DR attivo — nessun NaN, joint_vel_explosion non aumenta > 5%
   6. Terrains: viewer con ogni scene XML — robot non compenetra geometria

   ---
   Dipendenza API Critica da Verificare                                                                             
   Prima di implementare l'Assistance Curriculum, verificare quale API IsaacLab è disponibile:
   python -c "from isaaclab.assets import Articulation; print(dir(Articulation))" | grep -i force
 Cercare: apply_external_force_and_torque, external_force_b, o set_external_force.