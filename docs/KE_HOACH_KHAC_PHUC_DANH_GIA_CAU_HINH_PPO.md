# Kế hoạch khắc phục các khuyết điểm của cấu hình PPO quadcopter

## 1. Mục tiêu và phạm vi

Kế hoạch này chuyển các nhận xét trong
`docs/DANH_GIA_CAU_HINH_PPO_QUADCOPTER.md` thành các hạng mục triển khai,
kiểm thử và đánh giá có thứ tự ưu tiên. Phạm vi gồm:

- `velocity_ned`: baseline điều khiển bằng ba velocity setpoint NED;
- `motor_residual`: phương án điều khiển trực tiếp bốn rotor;
- PPO dùng chung, reward, episode semantics, curriculum, monitoring,
  checkpoint và quy trình đánh giá.

Kế hoạch không mặc định rằng mọi đề xuất trong bản đánh giá đều còn đúng với
checkout hiện tại. Mỗi hạng mục phải được đối chiếu lại với source và cấu hình
trước khi sửa.

## 2. Kết quả đối chiếu checkout hiện tại

### 2.1. Các điểm đã được giải quyết hoặc đã có bằng chứng trong code

- `velocity_ned` thực sự gửi velocity setpoint bằng
  `move_by_velocity_async(v_north, v_east, v_down, ...)`; action không phải RPM
  hoặc thrust.
- Motor action có thứ tự `[FL, FR, RL, RR]`, dùng mapping
  `hover_centered_full_signal` version 2:
  `action=-1/0/+1` tương ứng `control_signal=0/hover/1`.
- Motor policy dùng `tanh_squashed_gaussian`; environment từ chối action ngoài
  `[-1,1]` thay vì tự clip.
- Reward motor có công thức cụ thể, scale theo `dt/reference_dt` và log được các
  thành phần reward.
- Curriculum motor có bảy stage, ngưỡng chuyển stage và domain randomization.
- Checkpoint motor có sidecar metadata và từ chối mapping version cũ.

Các nội dung trên cần được giữ làm invariant; không lập lại hoặc thay đổi nếu
không có giả thuyết thực nghiệm rõ ràng.

### 2.2. Các sai lệch hoặc khuyết điểm xác nhận từ source

| ID | Mức độ | Hiện trạng xác nhận | Hậu quả |
|---|---|---|---|
| C-01 | Rất cao | `velocity_ned.train` chỉ lặp 50 bước nhưng environment có time limit khác; nếu đủ 50 bước mà chưa `done`, bước cuối không được đánh dấu kết thúc episode | Return có thể nối sai qua ranh giới reset |
| C-02 | Rất cao | PPO dùng một cờ `is_terminals`; training gộp `terminated or truncated` | Time-limit truncation bị xử lý như terminal thật, không bootstrap value |
| C-03 | Rất cao | `success bonus` của `velocity_ned` được cộng lặp sau 20 bước và lệnh kết thúc success đang bị comment | Có nguy cơ reward hacking và không có success event rõ ràng |
| C-04 | Cao | `velocity_ned` không phạt rõ collision/attitude-limit trong mọi đường termination | Reward cuối có thể không phản ánh thất bại vật lý |
| C-05 | Cao | Vòng chờ pose của `velocity_ned` không có timeout | Training có thể treo vô hạn khi simulator không phản hồi |
| C-06 | Cao | Log velocity chỉ có average reward; log motor chưa có PPO loss/KL/clip fraction/value diagnostics | Không đủ dữ liệu để đánh giá update stability |
| C-07 | Cao | Goal motor thay đổi theo `target_z_range_m`, nhưng observation chỉ có absolute position và scalar distance | Policy không quan sát trực tiếp hướng sai số mục tiêu khi goal thay đổi |
| C-08 | Cao | Curriculum motor chỉ random XY tối đa 0.4 m và chủ yếu thay đổi altitude | Chưa đủ bằng chứng policy học target reaching từ miền vị trí rộng |
| C-09 | Trung bình | `velocity_ned.ACTION_STD_INIT` hiện là 1.5, trong khi tài liệu đánh giá dùng 0.5; biến thể short episode dùng 0.5 | Config/document drift, kết luận exploration có thể sai |
| C-10 | Trung bình | PPO chưa xuất approximate KL, clip fraction, explained variance; `K_EPOCHS=30` chưa có early stop theo KL | Không phát hiện update quá mạnh hoặc critic kém |
| C-11 | Trung bình | Motor reward position dạng mũ nhưng không có progress term tường minh | Tín hiệu học có thể yếu khi khởi tạo xa goal |
| C-12 | Trung bình | Chưa có success state machine tách `reached`, `sustained hover`, `failure`, `time limit` | Không đo được hai giai đoạn nhiệm vụ độc lập |

## 3. Ràng buộc bắt buộc của kế hoạch

### 3.1. Ràng buộc an toàn và vận hành

1. Không tự động khởi động training, kết nối simulator, arm drone hoặc gửi
   motor command trong unit test/CI.
2. Mọi smoke test, calibration và simulator evaluation phải mặc định dry-run;
   thao tác thật cần cờ `--execute` hoặc lệnh riêng rõ ràng.
3. Không overwrite checkpoint, calibration, log hoặc kết quả thí nghiệm.
   Mỗi run phải có `run_id` mới và snapshot cấu hình riêng.
4. Motor signal gửi ProjectAirSim phải thuộc `[0,1]`; residual action phải thuộc
   `[-1,1]`. Action không hợp lệ phải fail-fast, không âm thầm clip.
5. Thứ tự actuator luôn là `[FL, FR, RL, RR]` theo actuator ID; không dựa vào
   thứ tự khai báo nội bộ của controller.
6. Giới hạn slew-rate motor chỉ được chọn sau khi đo/đối chiếu actuator dynamics;
   không tự đặt một hằng số chưa có bằng chứng.

### 3.2. Ràng buộc kiến trúc và tương thích

1. Giữ ranh giới module: thay đổi `motor_residual` không được làm thay đổi hành
   vi `velocity_ned` nếu chưa có hạng mục được phê duyệt cho cả hai.
2. Trước khi sửa PPO dùng chung, phải kiểm tra callers/impact và chạy test cho
   cả `(13,3)` velocity lẫn motor configuration mới.
3. Mọi thay đổi semantics của state, action, reward, termination hoặc policy
   distribution phải tăng version tương ứng trong checkpoint metadata.
4. Checkpoint cũ không được resume nếu version semantics không khớp, kể cả khi
   tensor shape vẫn giống nhau.
5. Không thay đồng thời observation, reward và PPO hyperparameter trong cùng
   một thí nghiệm so sánh; nếu cần thay đổi tích hợp, phải có ablation riêng.

### 3.3. Ràng buộc khoa học và tái lập

1. Chốt task contract và success thresholds trước khi xem kết quả đánh giá;
   không điều chỉnh hậu nghiệm để tăng success rate.
2. Tách development scenarios và frozen evaluation scenarios bằng seed và file
   cấu hình khác nhau.
3. Mỗi run phải lưu seed, source fingerprint, full hyperparameters, task
   contract version, observation/reward/action mapping version và curriculum.
4. Development có thể dùng tối thiểu 3 training seeds; kết quả báo cáo cuối
   dùng tối thiểu 5 seeds nếu tài nguyên cho phép. Báo cáo từng seed, mean,
   standard deviation và confidence interval phù hợp.
5. Reward chỉ là tín hiệu huấn luyện, không phải tiêu chí kết luận. Đánh giá bắt
   buộc dùng metrics vật lý và success state machine.
6. RMSE của episode kết thúc sớm phải đi kèm completion/survival/crash và elapsed
   time; không dùng RMSE trên executed prefix để che phần nhiệm vụ chưa hoàn tất.

## 4. Task contract cần đóng băng trước khi sửa thuật toán

Tạo một cấu hình versioned dùng chung cho evaluation. Giá trị dưới đây là
baseline đề xuất và phải được chốt trước khi chạy thí nghiệm chính thức:

| Thuộc tính | Baseline đề xuất |
|---|---|
| Coordinate convention | NED; `Down > 0`, bay lên có `z < 0` |
| Goal nominal | `[0, 0, -2] m` |
| Goal policy | Cố định cho velocity baseline; thay đổi có kiểm soát cho motor |
| Yaw target | `0 rad` trong baseline; nếu yaw không thuộc nhiệm vụ thì bỏ cả yaw penalty và yaw success constraint |
| Reach condition | position error `< 0.20 m` |
| Hover velocity | speed `< 0.20 m/s` |
| Hover attitude | norm roll-pitch `< 10 deg` |
| Training sustained duration | 2 s liên tục, dùng để tạo success event |
| Evaluation sustained duration | 5 s liên tục; thêm stress test 10 s |
| Failure | collision, ground touch, attitude/velocity/position safety violation |
| Time limit | `truncated`, không phải physical failure |

Các ngưỡng trên kế thừa điều kiện reward hiện có để tạo baseline nhất quán. Nếu
đổi ngưỡng, phải tăng `task_contract_version` và không so sánh trực tiếp success
rate giữa hai version.

## 5. Các giai đoạn triển khai

### Giai đoạn 0 — Đóng băng specification và nguồn cấu hình

**Mục tiêu:** loại bỏ mâu thuẫn giữa code, bảng cấu hình và bản đánh giá.

#### Công việc

- Tạo `TaskContract`/file cấu hình versioned cho goal, yaw policy, reach/hover
  thresholds, sustained duration, safety thresholds và time limit.
- Chọn một nguồn cấu hình duy nhất cho từng action space; tài liệu sinh hoặc
  kiểm tra từ nguồn này thay vì lặp lại số liệu thủ công.
- Ghi rõ `velocity_ned` action là velocity setpoint NED với đơn vị m/s.
- Sửa sai lệch `ACTION_STD_INIT` giữa `velocity_ned.train`, short-episode và
  tài liệu; không quyết định 0.5 hay 1.5 chỉ bằng nhận xét lý thuyết.
- Thêm `task_contract_version`, `observation_schema_version`,
  `reward_version` và PPO configuration vào metadata run/checkpoint.

#### Tiêu chí hoàn thành

- Không còn hai giá trị khác nhau cho cùng một profile cấu hình.
- Unit test xác nhận NED signs, units, action bounds và goal/yaw semantics.
- Tài liệu nêu rõ profile nào áp dụng cho entry point nào.

### Giai đoạn 1 — Sửa correctness của episode và PPO buffer

**Mục tiêu:** bảo đảm return/advantage đúng trước khi tuning.

#### Công việc

1. Dùng một nguồn time limit duy nhất giữa environment và training loop; không
   để `for range(STEPS_PER_EPISODE)` kết thúc episode mà không phát cờ.
2. Tách rõ:
   - `terminated`: failure hoặc task success theo MDP;
   - `truncated`: hết time limit/giới hạn bên ngoài.
3. Thay buffer terminal boolean bằng masks đủ để bootstrap value ở truncation.
4. Triển khai return/advantage có kiểm thử; ưu tiên GAE-Lambda có cấu hình
   (`gamma`, `gae_lambda`) và bootstrap từ final observation khi truncated.
5. Thêm timeout cho pose synchronization của `velocity_ned`; timeout phải dừng
   điều khiển an toàn và ghi failure reason.
6. Chuẩn hóa `info` với `termination_reason`, `is_success`, `time_limit`,
   `final_observation` và metrics cuối episode.

#### Kiểm thử bắt buộc

- Episode time-limit không nối return sang episode sau.
- Physical terminal có bootstrap bằng 0; truncation bootstrap từ critic.
- Success, collision, ground touch và timeout tạo reason khác nhau.
- Regression test cho cả training loop velocity và motor.

#### Gate

Không thực hiện hyperparameter sweep trước khi Giai đoạn 1 đạt toàn bộ test.

### Giai đoạn 2 — Success state machine và reward correctness

**Mục tiêu:** reward hỗ trợ nhiệm vụ nhưng success được xác định độc lập.

#### Công việc chung

- Tạo state machine:
  `approaching → reached → sustaining → success`, có reset bộ đếm khi vi phạm
  bất kỳ điều kiện hover nào.
- Log `reached_once`, `time_to_target`, `sustained_seconds`, `success`,
  `overshoot`, và reason nếu thất bại.
- Success bonus chỉ phát một lần theo event hoặc được giới hạn rõ; không cộng
  lặp vô hạn sau ngưỡng.
- Log từng reward component với units và giá trị trước/sau weighting.
- Tạo script/fixture vẽ reward surface theo distance, velocity, attitude,
  angular rate và motor signal mà không cần simulator.

#### Riêng `velocity_ned`

- Sửa success bonus lặp và quyết định rõ success có terminate training episode
  hay tiếp tục một cửa sổ sustain cố định.
- Bổ sung penalty thống nhất cho collision, ground, attitude-limit và
  velocity-limit.
- Công bố chính xác `r_pos`, `r_approach`, attitude penalty và điều kiện bonus.
- Xác minh `r_approach` không phụ thuộc sai vào state sau reset.

#### Riêng `motor_residual`

- Giữ reward v1 làm baseline và kiểm tra reward curves trên miền curriculum.
- Chỉ bổ sung progress shaping nếu position reward thực sự yếu ở miền xa.
  Ưu tiên potential-based shaping và giới hạn magnitude so với crash/hover
  terms; tạo ablation `reward_v1` so với `reward_v2_progress`.
- Chốt yaw policy: nếu yaw target là 0, thêm yaw vào success metrics; nếu yaw
  không quan trọng, loại yaw penalty thay vì giữ ràng buộc ngầm.

#### Tiêu chí hoàn thành

- Một episode chỉ có tối đa một success event/bonus sự kiện.
- Reward unit tests bao phủ near/far goal, approaching/receding, stable/unstable,
  crash và boundary conditions.
- Success metrics không được suy ra từ tổng reward.

### Giai đoạn 3 — Observation và goal representation

**Mục tiêu:** loại bỏ partial observability bằng cách đồng bộ observation với
phân bố goal được dùng trong huấn luyện.

#### Quyết định thiết kế

- `velocity_ned` baseline giữ goal cố định và state 13 chiều để làm control.
- Theo cấu hình người dùng chọn, `motor_residual` dùng observation v3 gồm đúng
  các chỉ số 0–12, tạo state 13 chiều.
- Vì state 13 không chứa vector `[dx,dy,dz]`, curriculum mặc định phải giữ goal
  cố định `[0,0,-2]`; reaching được tạo bằng cách random initial state. Nếu
  khôi phục variable goal, phải bổ sung goal direction vào observation.

#### Công việc

- Chuẩn hóa và tài liệu hóa units/ranges cho từng observation field.
- Đánh giá running normalization hoặc fixed scaling; `LayerNorm` không được xem
  là bằng chứng duy nhất rằng input distribution đã phù hợp.
- Dùng observation schema version 3 và từ chối checkpoint state 17/20 cũ.
- Tạo test cho thứ tự chính xác 13 trường, NED convention, khoảng cách tới goal
  và sensor noise.

#### Gate

Không dùng checkpoint motor state 17/20 để kết luận hiệu năng của observation v3.

### Giai đoạn 4 — Horizon và curriculum target reaching

**Mục tiêu:** tạo đủ thời gian và phân bố reset để học cả reaching lẫn hover.

#### `velocity_ned`

- So sánh horizon vật lý 10 s, 15 s và 20 s; giữ `dt=0.1 s` trong thí nghiệm
  horizon để không trộn ảnh hưởng tần số điều khiển.
- Chọn horizon theo phân bố `time_to_target + sustained_duration`, không chọn
  chỉ vì reward lớn hơn.

#### `motor_residual`

- Giữ các stage hover gần hiện tại làm safety curriculum.
- Thêm stage reaching bằng cách tăng dần miền random initial position quanh
  goal cố định; không random goal khi state không có vector hướng tới goal.
- Tách training distribution và evaluation distribution. Evaluation phải có
  initial positions chưa xuất hiện chính xác trong training, nhưng vẫn dùng
  cùng goal `[0,0,-2]`.
- Chỉ mở rộng position limit 5 m sau khi định nghĩa operational envelope và
  kiểm chứng safety; không bỏ termination chỉ để tăng completion.
- Xác nhận disturbance là transient hay sustained; external force phải có
  duration và thời điểm áp dụng được log rõ ràng.

#### Tiêu chí chuyển stage đề xuất

- Dùng đồng thời survival rate, reach success rate, sustained-hover success,
  position RMSE và attitude RMSE.
- Cửa sổ phải có số episode tối thiểu và không tái sử dụng episode của stage
  trước sau khi advance.
- Không advance chỉ nhờ reward mean.

### Giai đoạn 5 — Instrumentation PPO và hyperparameter experiments

**Mục tiêu:** tuning dựa trên diagnostics thay vì phỏng đoán.

#### Instrumentation cần bổ sung

- policy loss, value loss, entropy/action std;
- approximate KL và clip fraction theo epoch/update;
- explained variance và value error;
- gradient norm actor/critic;
- action mean/std theo chiều và tỷ lệ gần biên;
- transitions/update, episode mix và stage distribution.

#### Cơ chế bảo vệ update

- Cấu hình `target_kl` và early-stop epoch khi KL vượt ngưỡng đã chốt.
- Không mặc định `K_EPOCHS=30`; so sánh 10, 20 và 30 sau khi có KL/clip logs.
- Kiểm tra reward/advantage normalization trên batch nhỏ và batch có nhiều
  episode.
- Không đổi actor LR, critic LR, gamma và epochs trong cùng một comparison.

#### Ma trận thí nghiệm tối thiểu

1. Baseline đã sửa correctness.
2. `K_EPOCHS`: 10/20/30, giữ các biến khác cố định.
3. Motor `gamma`: 0.99/0.995/0.999, đánh giá effective horizon, critic stability
   và variance; không chọn gamma chỉ dựa vào final reward.
4. Action-std schedule: log schedule hiện tại trước, sau đó mới so sánh decay
   tuyến tính theo episode với schedule theo timestep/update.
5. Observation/reward ablations chạy tách biệt.

### Giai đoạn 6 — Bộ đánh giá độc lập

#### Test A: Hover gần cân bằng

- Reset gần goal với perturbation nhỏ.
- Đo sustained-hover success, position/velocity/attitude/angular-rate RMSE,
  control effort và saturation.

#### Test B: Target reaching

- Reset ở nhiều vị trí, độ cao, vận tốc và attitude.
- Đo reach success, time-to-target, path length, overshoot, crash và timeout.

#### Test C: Reaching rồi sustained hovering

- Bắt đầu ngoài vùng goal, yêu cầu vào vùng và duy trì 5 s; stress test 10 s.
- Episode phải đủ dài để chứa cả reaching và sustain window.

#### Test D: Robustness/recovery

- Mass/inertia, thrust effectiveness, hover bias, motor lag, sensor noise,
  action delay và wind/gust theo từng nhóm riêng.
- Báo cáo recovery rate/time, peak error, crash rate và saturation.
- Có test interpolation trong miền training và extrapolation ngoài miền
  training nhưng vẫn trong safety envelope.

#### Metrics bắt buộc

| Nhóm | Metrics |
|---|---|
| Reaching | reach success, time-to-target, overshoot, path efficiency |
| Hover | sustained success, position/velocity/attitude/angular-rate RMSE |
| Safety | crash, ground touch, attitude/velocity/position limit, timeout |
| Control | effort, smoothness, saturation fraction, action near-bound fraction |
| PPO | KL, clip fraction, entropy/std, explained variance, losses |
| Robustness | recovery rate/time, peak deviation, post-disturbance RMSE |

### Giai đoạn 7 — Tài liệu hóa và release gate

#### Công việc

- Đồng bộ `PPO_CONFIG_TABLE.md`, usage guide, module README và evaluation
  protocol với config version thực tế.
- Mỗi bảng kết quả ghi rõ task contract, state/action/reward versions, seed,
  checkpoint metadata và evaluation scenario version.
- Tạo migration note: checkpoint nào tương thích, checkpoint nào chỉ dùng để
  tham khảo lịch sử.

#### Release gate

Một cấu hình chỉ được gọi là “giải quyết reaching và sustained hovering” khi:

1. Pass Test A, B và C trên frozen evaluation set.
2. Có kết quả nhiều seed và không chỉ chọn best checkpoint/best seed.
3. Không có config/document drift.
4. Metrics safety và completion được báo cáo cùng RMSE/reward.
5. Test, compile, `git diff --check` và CodeGraph sync đều đạt.
6. Simulator evaluation có manifest và không overwrite artifact cũ.

## 6. Thứ tự ưu tiên và phụ thuộc

```text
Giai đoạn 0: Task contract/config truth
        ↓
Giai đoạn 1: Episode + bootstrap correctness
        ↓
Giai đoạn 2: Success/reward correctness
        ↓
Giai đoạn 3: Observation v2 (motor)
        ↓
Giai đoạn 4: Horizon + reaching curriculum
        ↓
Giai đoạn 5: PPO diagnostics + tuning
        ↓
Giai đoạn 6: Frozen evaluation + robustness
        ↓
Giai đoạn 7: Documentation + release gate
```

Giai đoạn 1 và 2 là blocker của mọi kết luận thực nghiệm. Giai đoạn 5 không được
đẩy lên trước chỉ vì tuning dễ chạy hơn sửa episode semantics.

## 7. Danh sách deliverable dự kiến

| Deliverable | Nội dung |
|---|---|
| Task contract | Goal/yaw/success/failure/time-limit versioned config |
| PPO buffer v2 | Tách termination/truncation, bootstrap đúng, GAE tests |
| Success metrics | Reached/sustained/success state machine dùng chung |
| Velocity env v2 | Timeout, reward events, horizon nhất quán, richer logging |
| Motor observation v3 | State 13 (chỉ số 0–12), goal cố định, schema/version và checkpoint guard |
| Curriculum v2 | Reaching stages và frozen evaluation distributions |
| PPO diagnostics | KL, clip fraction, entropy, value/gradient metrics |
| Evaluation runner | Test A–D, multi-seed manifest, aggregate report |
| Documentation | Config table, protocol, migration/compatibility note |

## 8. Định nghĩa hoàn thành toàn kế hoạch

Kế hoạch được xem là hoàn thành khi có thể trả lời bằng dữ liệu tái lập được:

1. Policy có đi vào vùng mục tiêu từ các điều kiện khởi tạo độc lập không?
2. Sau khi đến goal, policy có duy trì đồng thời vị trí, vận tốc và attitude đủ
   thời gian quy định không?
3. Kết quả có ổn định qua seed và không phụ thuộc một checkpoint được chọn lọc
   hậu nghiệm không?
4. Policy thất bại ở đâu dưới disturbance/model mismatch và có phục hồi không?
5. PPO update có nằm trong miền KL/clip/value stability đã định nghĩa không?

Chỉ sau khi năm câu hỏi này có bằng chứng đầy đủ mới chuyển từ “cấu hình có
tiềm năng” sang kết luận về khả năng giải quyết bài toán.

## 9. Trạng thái triển khai

Cập nhật ngày 2026-11-15:

| Giai đoạn | Trạng thái | Bằng chứng triển khai |
|---|---|---|
| 0 | Đã triển khai phần code/config | `drone_control/task_contract.py`; task contract v1, reward/observation/PPO config trong metadata motor |
| 1 | Đã triển khai và có unit test | Buffer tách `terminated`/`truncated`, GAE bootstrap time limit, horizon velocity 15 s, pose timeout |
| 2 | Đã triển khai lõi | Success tracker dùng chung, success event một lần, chuẩn hóa `info`; reward surface trực quan vẫn chưa tạo |
| 3 | Đã triển khai và có checkpoint guard | Motor observation v3 dùng chỉ số 0–12, state 13, metadata schema v4 từ chối checkpoint state 17/20 |
| 4 | Đã triển khai curriculum; chưa chạy so sánh horizon | Giữ goal `[0,0,-2]`, tăng độ khó bằng random initial state và success-rate gate; chưa có dữ liệu 10/15/20 s |
| 5 | Đã triển khai instrumentation | KL, clip fraction, explained variance, gradient norm, target-KL early stop và action near-bound trong log motor; chưa chạy sweep |
| 6 | Đã đóng băng protocol/metrics, chưa chạy simulator | `drone_control/evaluation_protocol.py`, `tools/create_evaluation_manifest.py`; chưa có kết quả Test A–D |
| 7 | Đã đồng bộ tài liệu kỹ thuật chính; release gate chưa đạt | `PPO_CONFIG_TABLE.md` phản ánh state 13/schema v4; cần dữ liệu multi-seed và simulator manifest hoàn tất |

Các mục “chưa chạy” là chủ ý: kế hoạch cấm tự động khởi động training hoặc
ProjectAirSim. Do đó thay đổi source có thể hoàn tất và kiểm thử offline, nhưng
không được đánh dấu toàn bộ kế hoạch là hoàn thành trước khi có kết quả thực
nghiệm nhiều seed theo Giai đoạn 5–7.
