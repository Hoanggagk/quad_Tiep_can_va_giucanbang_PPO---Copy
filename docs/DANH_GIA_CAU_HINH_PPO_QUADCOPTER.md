# ĐÁNH GIÁ KHOA HỌC CẤU HÌNH PPO CHO BÀI TOÁN QUADCOPTER TIẾP CẬN VÀ DUY TRÌ CÂN BẰNG TẠI ĐIỂM MỤC TIÊU

## 1. Mục tiêu và phạm vi đánh giá

Tài liệu này đánh giá cấu hình PPO được mô tả trong
`PPO_CONFIG_TABLE.md` đối với bài toán điều khiển quadcopter có hai yêu
cầu liên tiếp:

1.  **Target reaching:** quadcopter phải di chuyển từ trạng thái ban đầu
    đến vị trí mục tiêu.
2.  **Stable hovering:** sau khi đến vùng mục tiêu, quadcopter phải duy
    trì vị trí, vận tốc nhỏ và tư thế ổn định trong một khoảng thời gian
    đủ dài.

Đánh giá dựa trên cấu hình được cung cấp, kết hợp với phân tích lý
thuyết điều khiển và học tăng cường. Đây là đánh giá thiết kế cấu hình,
không phải xác nhận thực nghiệm rằng policy đã hội tụ hoặc có thể bay ổn
định. Muốn kết luận về hiệu năng cần có mã triển khai đầy đủ, định nghĩa
chính xác các hàm reward/termination, phân bố trạng thái khởi tạo và kết
quả đánh giá độc lập.

## 2. Kết luận tổng quan

**Kết luận: cấu hình có nền tảng phù hợp để nghiên cứu bài toán, nhưng
chưa đủ bằng chứng và còn một số điểm cần làm rõ hoặc điều chỉnh trước
khi khẳng định giải quyết tốt cả Target Reaching và Stable Hovering.**

Hai nhánh có đặc điểm khác nhau:

  -----------------------------------------------------------------------------
  Tiêu chí                `velocity_ned`                `motor_residual`
  ----------------------- ----------------------------- -----------------------
  Policy điều khiển       Ba vận tốc NED                Bốn residual action cho
                          `[v_north, v_east, v_down]`   bốn động cơ

  State                   13 chiều                      17 chiều

  Action                  3 chiều                       4 chiều

  Mức điều khiển          Mức vận tốc, phụ thuộc bộ     Mức actuator/motor,
                          điều khiển thấp hơn của       trực tiếp hơn
                          simulator                     

  Reward có thành phần vị Có `r_pos` và `r_approach`    Có position reward
  trí                                                   

  Reward duy trì hover    Có hover bonus và success     Có hover bonus
                          bonus                         

  Hạn chế nổi bật         Episode chỉ 5 giây; cần làm   Cần xác nhận khả năng
                          rõ cách tính reward và tín    tiếp cận mục tiêu từ
                          hiệu điều khiển vận tốc       các trạng thái khởi
                                                        tạo; thiếu mô tả rõ
                                                        ràng về phần thưởng
                                                        tiến triển đến mục tiêu
                                                        và tiêu chí duy trì
                                                        thành công

  Nhận định               Phù hợp để phát triển bộ điều Có tiềm năng học điều
                          khiển tiếp cận/hover ở mức    khiển tổng thể ở mức
                          setpoint vận tốc              motor nhưng khó tối ưu
                                                        và đòi hỏi hiệu chuẩn,
                                                        curriculum, đánh giá
                                                        chặt chẽ
  -----------------------------------------------------------------------------

Không nên kết luận `motor_residual` tốt hơn chỉ vì policy điều khiển
trực tiếp bốn động cơ. Điều khiển motor-level có quyền điều khiển chi
tiết hơn nhưng cũng có không gian hành động phức tạp hơn, nhạy với mô
hình động lực học và hiệu chuẩn actuator.

## 3. Đánh giá cấu hình `velocity_ned`

### 3.1. State và action space

Cấu hình sử dụng:

-   `STATE_DIM = 13`: vị trí, góc Euler, vận tốc tuyến tính, vận tốc góc
    và khoảng cách đến mục tiêu.
-   `ACTION_DIM = 3`: `[v_north, v_east, v_down]`.
-   Action range: `[-1.5, 1.5]`.
-   Goal pose: `[0, 0, -2]` theo hệ tọa độ NED.

**Điểm tích cực**

1.  Ba thành phần vận tốc NED là lựa chọn tương đối trực quan cho nhiệm
    vụ di chuyển đến điểm mục tiêu.
2.  State có cả vị trí, vận tốc và tư thế, do đó policy có thông tin để
    học cách giảm sai số vị trí, giảm vận tốc khi đến gần đích và hạn
    chế nghiêng.
3.  Nếu mục tiêu luôn cố định tại `[0, 0, -2]`, các tọa độ vị trí tuyệt
    đối có thể giúp policy suy ra sai số tương đối.

**Vấn đề cần kiểm tra**

-   Cần xác nhận action được simulator hiểu là vận tốc đặt (velocity
    setpoint), vận tốc giới hạn hay một dạng lệnh khác. Cách diễn giải
    này quyết định trực tiếp ý nghĩa của action range.
-   Cần xác nhận thứ tự trục và dấu của NED xuyên suốt state, goal và
    action. Trong NED, trục Down dương hướng xuống; mục tiêu có `z = -2`
    cần được diễn giải theo đúng quy ước của môi trường.
-   Nếu mục tiêu thay đổi theo episode, chỉ cung cấp khoảng cách
    `dist_to_goal` là chưa đủ để xác định hướng cần di chuyển. Nên đưa
    vector sai số mục tiêu vào observation, ví dụ `dx, dy, dz` trong hệ
    tọa độ nhất quán.
-   Cần kiểm tra việc chuẩn hóa state. Các đại lượng mét, m/s, radian và
    rad/s có thang giá trị khác nhau; `LayerNorm` hỗ trợ chuẩn hóa nội
    bộ nhưng không thay thế việc kiểm tra phân bố đầu vào và ngoại lệ.

**Khuyến nghị:** nếu mục tiêu cố định, có thể giữ cấu trúc hiện tại để
làm baseline. Nếu mục tiêu thay đổi, nên ưu tiên observation tương đối
với mục tiêu thay vì chỉ dựa vào tọa độ tuyệt đối và khoảng cách vô
hướng.

### 3.2. Action standard deviation

Các giá trị được mô tả:

-   `ACTION_STD_INIT = 0.5`
-   `MIN_ACTION_STD = 0.01`
-   `ACTION_STD_DECAY_RATE = 0.01`
-   Decay mỗi 50 episode.

Mức khám phá ban đầu 0.5 có thể hợp lý để bắt đầu thử nghiệm, nhưng cần
đánh giá theo phân bố action thực tế sau khi policy xuất action và qua
bước scale. Không thể chỉ dựa vào tỷ lệ giữa standard deviation và
action range để kết luận exploration là đủ.

**Điểm cần làm rõ:** tài liệu ghi `MIN_ACTION_STD = 0.01` tương đương
khoảng `±0.15 m/s` khi khai thác. Nếu action được scale tuyến tính với
hệ số 1.5, standard deviation 0.01 tương ứng khoảng 0.015 m/s, không
phải 0.15 m/s. Cần kiểm tra lại phép quy đổi và phân biệt standard
deviation với khoảng biến thiên thường gặp của phân phối.

Ngoài ra, phải xác định `DECAY_RATE = 0.01` được cài đặt theo cách nào:
trừ một lượng cố định, nhân với hệ số, hay quy tắc khác. Ba cách này tạo
lịch exploration rất khác nhau. Nếu giảm quá nhanh, policy có thể mất
khả năng khám phá trước khi học được cách tiếp cận mục tiêu; nếu giảm
quá chậm, policy có thể khó ổn định khi hover.

**Khuyến nghị:** log standard deviation theo timestep/episode, action
trung bình, action thực tế và tỷ lệ action chạm giới hạn. Chỉ điều chỉnh
lịch decay sau khi quan sát các chỉ số này.

### 3.3. Training horizon và thời gian episode

Cấu hình:

-   `dt = 0.1 s`
-   `STEPS_PER_EPISODE = 50`
-   Thời lượng tối đa mỗi episode: `5 s`
-   `MAX_TRAINING_TIMESTEPS = 500,000`
-   `UPDATE_TIMESTEP = 2,000`

Episode 5 giây là điểm đáng lưu ý nhất của nhánh này.

Một quadcopter có thể cần thời gian để tăng tốc, di chuyển đến mục tiêu,
giảm tốc, triệt tiêu vận tốc dư và ổn định tư thế. Nếu khoảng cách ban
đầu lớn hoặc có giới hạn gia tốc, 5 giây có thể không đủ để thể hiện đầy
đủ cả quá trình tiếp cận và duy trì hover.

Với 50 bước/episode, điều kiện `hover thành công >20 bước` tương ứng hơn
2 giây nếu được tính theo từng bước 0.1 giây. Đây là thời gian duy trì
tương đối ngắn để chứng minh khả năng hover bền vững. Cần phân biệt đây
là điều kiện nhận thưởng hay điều kiện kết thúc episode.

**Khuyến nghị:**

-   Tăng horizon episode để policy có đủ thời gian tiếp cận và duy trì
    tại đích. Có thể thử nghiệm các mức 10, 15 và 20 giây, sau đó chọn
    dựa trên phân bố thời gian tiếp cận và độ ổn định.
-   Thiết kế giai đoạn đánh giá riêng, trong đó quadcopter phải duy trì
    hover liên tục ít nhất 5--10 giây sau khi vào vùng mục tiêu. Đây là
    đề xuất cho kiểm chứng, không phải giá trị bắt buộc cho mọi môi
    trường.
-   Không chỉ tăng episode length; cần kiểm tra trạng thái khởi tạo và
    khoảng cách ban đầu để bảo đảm bài toán thực sự đòi hỏi tiếp cận.

### 3.4. PPO hyperparameters

Cấu hình:

-   `EPS_CLIP = 0.2`
-   `GAMMA = 0.99`
-   `LR_ACTOR = 3e-4`
-   `LR_CRITIC = 1e-3`
-   `K_EPOCHS = 30`
-   Batch/update tối đa 2,000 timestep.

Đây là một bộ giá trị có thể dùng làm baseline PPO. Tuy nhiên, chưa có
cơ sở để coi đây là cấu hình tối ưu cho quadcopter.

**Các điểm cần lưu ý**

1.  `GAMMA = 0.99` tại `dt = 0.1 s` tương ứng thời hằng chiết khấu xấp
    xỉ 10 giây theo quy ước `dt/(1-gamma)`. Nó có thể phù hợp với mục
    tiêu dài hạn, nhưng episode hiện tại chỉ 5 giây nên horizon episode
    đang giới hạn khả năng học nhiều hơn bản thân gamma.
2.  `K_EPOCHS = 30` là mức tối ưu lặp tương đối cao trên cùng rollout.
    Có nguy cơ policy cập nhật quá mức hoặc overfit batch, nhất là khi
    dữ liệu tương quan theo thời gian. Cần theo dõi approximate KL, clip
    fraction, entropy, policy loss và value loss.
3.  `LR_CRITIC` lớn hơn `LR_ACTOR` có thể giúp critic thích nghi nhanh,
    nhưng cũng có thể làm value function dao động nếu reward có thang
    lớn hoặc thay đổi mạnh. Cần theo dõi explained variance và lỗi
    value.
4.  `EPS_CLIP = 0.2` là lựa chọn phổ biến, không phải bằng chứng về độ
    ổn định của policy.

**Khuyến nghị thử nghiệm:** giữ cấu hình hiện tại làm baseline; sau đó
so sánh có kiểm soát `K_EPOCHS` thấp hơn (ví dụ 10--20), learning rate
và gamma. Mỗi lần chỉ thay đổi một nhóm tham số, chạy nhiều random seed
và báo cáo trung bình cùng độ lệch chuẩn.

### 3.5. Reward function

Reward được mô tả gồm:

-   `r_pos`: trọng số 1.0.
-   `r_approach`: trọng số 2.0.
-   `r_att`: trọng số -0.1.
-   `survival bonus`: 0.01.
-   `hover bonus`: 0.5.
-   `success bonus`: 5.0 khi hover thành công hơn 20 bước.
-   `crash penalty`: -2.0.
-   `ground penalty`: -2.0.

**Điểm tích cực**

-   Có cả thành phần hướng đến mục tiêu, ổn định tư thế và duy trì
    hover.
-   Có thưởng thành công và phạt va chạm/chạm đất.
-   Reward có định hướng phù hợp với cấu trúc nhiệm vụ gồm tiếp cận rồi
    duy trì.

**Vấn đề quan trọng**

Bảng mới chỉ mô tả tên, trọng số và ý nghĩa khái quát; chưa có công thức
chính xác của từng thành phần. Do đó chưa thể đánh giá gradient reward,
độ lớn tương đối và khả năng xảy ra reward hacking.

Cần làm rõ:

-   `r_pos` giảm theo hàm nào và có phụ thuộc khoảng cách mục tiêu hay
    không?
-   `r_approach` thưởng theo khoảng cách giảm giữa hai bước, theo vận
    tốc hướng về đích, hay theo tiêu chí khác?
-   `r_att` phạt roll/pitch hay cả yaw? Có chuẩn hóa góc và xử lý góc
    wrap-around không?
-   `hover bonus` được cấp mỗi bước hay chỉ khi bắt đầu đạt điều kiện?
-   `success bonus` được cấp một lần hay lặp lại ở mọi bước sau khi đạt
    ngưỡng?
-   Điều kiện `<0.2 m` và `<0.2 m/s` có được duy trì liên tục hay chỉ
    cần đạt tại một thời điểm?

Đặc biệt, nếu `success bonus` được cộng lặp ở mọi bước sau khi đạt 20
bước, cần bảo đảm episode không thể khai thác phần thưởng bằng cách kéo
dài thời gian mà không thực sự duy trì ổn định theo định nghĩa mong
muốn.

**Khuyến nghị:** công bố công thức đầy đủ và log từng thành phần reward
độc lập. Xây dựng điều kiện thành công theo dạng đồng thời: - sai số vị
trí dưới ngưỡng; - tốc độ tuyến tính dưới ngưỡng; - roll/pitch dưới
ngưỡng; - các điều kiện trên được duy trì liên tục trong thời gian quy
định.

### 3.6. Nhận định đối với `velocity_ned`

`velocity_ned` có cấu trúc state-action và reward ban đầu phù hợp với
bài toán tiếp cận mục tiêu và hover. Tuy nhiên, episode 5 giây và thời
gian xác nhận hover hơn 2 giây có thể chưa đủ để chứng minh khả năng duy
trì ổn định. Cần kiểm tra kỹ công thức reward, semantics của action và
cách reset episode.

**Đánh giá:** phù hợp làm baseline cấp vận tốc; chưa đủ căn cứ để khẳng
định đáp ứng đầy đủ bài toán nếu chưa tăng/kiểm chứng thời gian duy trì
và đánh giá trên nhiều điều kiện khởi tạo.

## 4. Đánh giá cấu hình `motor_residual`

### 4.1. State space 17 chiều

State gồm:

-   Vị trí NED: `x, y, z`.
-   Tư thế: `roll, pitch, yaw`.
-   Vận tốc tuyến tính: `vx, vy, vz`.
-   Vận tốc góc: `roll_rate, pitch_rate, yaw_rate`.
-   Khoảng cách đến mục tiêu.
-   Tín hiệu điều khiển hiện tại của bốn motor.

**Điểm tích cực**

1.  State có đủ nhóm đại lượng cơ bản để quan sát chuyển động tịnh tiến
    và quay của quadcopter.
2.  Motor signals hiện tại cung cấp thông tin về operating point của
    actuator.
3.  Có thể giúp policy học cách điều chỉnh motor quanh trạng thái hover
    đã hiệu chuẩn.

**Vấn đề cần kiểm tra**

-   Tương tự `velocity_ned`, nếu goal thay đổi, cần observation biểu
    diễn vector sai số tới goal, không chỉ khoảng cách vô hướng.
-   Cần xác nhận yaw mục tiêu. Reward hiện phạt sai lệch yaw về 0; điều
    này chỉ đúng nếu yaw = 0 thực sự là yêu cầu của bài toán.
-   Cần xác nhận motor signal trong observation có cùng định nghĩa, thứ
    tự và scale với tín hiệu được dùng trong action mapping.
-   Phải kiểm tra phân bố, giới hạn và chuẩn hóa từng biến. Motor signal
    thuộc `[0,1]`, góc tính radian, vị trí tính mét và vận tốc tính m/s.

### 4.2. Action space và motor mapping

Policy sinh bốn residual action `[FL, FR, RL, RR]` trong `[-1,1]`.
Action được ánh xạ quanh operating point hover đã hiệu chuẩn.

Đây là thiết kế hợp lý về mặt ý tưởng cho residual control: policy có
thể tăng/giảm tín hiệu từng rotor quanh mức cần thiết để hover, thay vì
phải học từ đầu toàn bộ mức ga.

**Ưu điểm**

-   Có operating point làm điểm xuất phát cho điều khiển hover.
-   Action chuẩn hóa giúp policy làm việc trong miền xác định.
-   `tanh-squashed Gaussian` giúp action nằm trong miền hợp lệ và có
    hiệu chỉnh Jacobian cho `log_prob`.
-   Có xét đến thứ tự actuator và metadata để tránh resume checkpoint
    sai mapping.

**Rủi ro và yêu cầu kiểm chứng**

-   Sai lệch nhỏ trong hiệu chuẩn hover signal có thể khiến policy phải
    bù liên tục hoặc không có đủ authority để giữ độ cao.
-   Cần kiểm tra sự phù hợp giữa `max_rpm`, thrust model, khối lượng,
    inertia, trọng tâm và operating point.
-   Bốn motor action độc lập có thể tạo ra tổ hợp thrust/torque không ổn
    định. Cần kiểm tra giới hạn tốc độ thay đổi, saturation và phản ứng
    của simulator.
-   Cần xác nhận simulator thực sự nhận `rpm_i` theo đúng quan hệ trong
    tài liệu và không có thêm actuator dynamics hoặc giới hạn tốc độ
    motor chưa được mô hình hóa.

**Khuyến nghị:** trước khi huấn luyện PPO, chạy thử operating point với
policy cố định `a=[0,0,0,0]` trong nhiều trạng thái ban đầu. Nếu cấu
hình hover đã hiệu chuẩn không thể giữ ổn định cơ bản, PPO sẽ phải học
đồng thời cả hiệu chỉnh nền và điều khiển, làm tăng độ khó đáng kể.

### 4.3. PPO hyperparameters và thời gian episode

Cấu hình:

-   `ACTION_STD_INIT = 0.35`
-   `MIN_ACTION_STD = 0.05`
-   `MAX_TRAINING_TIMESTEPS = 500,000`
-   `STEPS_PER_EPISODE = 500`
-   `dt = 0.02 s`
-   Episode tối đa 10 giây.
-   `UPDATE_TIMESTEP = 2,000`
-   `K_EPOCHS = 30`
-   `GAMMA = 0.99`
-   `EPS_CLIP = 0.2`
-   Actor LR `3e-4`, Critic LR `1e-3`.

So với `velocity_ned`, horizon 10 giây tốt hơn cho việc quan sát một chu
trình tiếp cận và duy trì hover. Tuy vậy, 10 giây vẫn có thể chưa đủ nếu
khoảng cách ban đầu lớn hoặc cần kiểm chứng duy trì ổn định dài hơn.

Với `GAMMA = 0.99` và `dt = 0.02 s`, thời hằng chiết khấu xấp xỉ 2 giây
theo quy ước `dt/(1-gamma)`. Điều này khiến reward tương lai xa bị giảm
tương đối nhanh theo thời gian mô phỏng. Với nhiệm vụ cần tiếp cận rồi
giữ ổn định trong nhiều giây, nên thử nghiệm gamma cao hơn, chẳng hạn
0.995 hoặc 0.999, nhưng phải đánh giá cùng độ ổn định của critic và độ
dài rollout.

Một PPO update nhận tối đa khoảng 2,000 transition, tương đương bốn
episode đủ 500 bước. `K_EPOCHS = 30` có nguy cơ tối ưu quá nhiều lần
trên batch tương quan. Nên theo dõi KL divergence, clip fraction,
entropy và value loss trước khi giữ nguyên.

Tổng 500,000 timestep ở 50 Hz tương ứng 10,000 giây mô phỏng tích lũy,
tức khoảng 2.78 giờ mô phỏng nếu chạy liên tục không tính overhead. Với
episode tối đa 500 bước, số episode tối đa theo trường hợp đủ horizon là
khoảng 1,000, không phải 10,000. Số thực tế phụ thuộc termination sớm.

**Khuyến nghị:** giữ các giá trị hiện tại làm baseline, đồng thời thử
gamma cao hơn và giảm số epoch tối ưu khi có dấu hiệu policy update quá
mạnh. Không tăng đồng thời nhiều tham số trong một thí nghiệm.

### 4.4. Reward function

Reward motor-level gồm:

-   Position: `position_weight = 1.0`, `position_scale = 0.5`, dạng
    `exp(-0.5 * position_error²)`.
-   Velocity: 0.10.
-   Attitude: 0.50.
-   Yaw: 0.05.
-   Angular rate: 0.05.
-   Motor effort: 0.02.
-   Motor smoothness: 0.10.
-   Saturation: 0.20.
-   Alive bonus: 0.01.
-   Hover bonus: 0.50.
-   Crash penalty: 10.0.

Điều kiện hover bonus: - position error `< 0.20 m`; - speed
`< 0.20 m/s`; - chuẩn attitude roll-pitch `< 10°`.

**Điểm tích cực**

1.  Reward không chỉ tối ưu vị trí mà còn xét vận tốc, attitude và
    angular rate.
2.  Motor effort và motor smoothness giúp hạn chế điều khiển quá mạnh
    hoặc thay đổi motor đột ngột.
3.  Saturation penalty khuyến khích policy tránh vận hành actuator sát
    biên.
4.  Hover bonus có điều kiện đồng thời về vị trí, vận tốc và attitude,
    phù hợp hơn với mục tiêu hover so với chỉ dựa vào khoảng cách.

**Vấn đề cần làm rõ**

-   Position reward dạng hàm mũ có thể tạo tín hiệu học yếu khi sai số
    vị trí lớn. Cần xem xét đường cong reward theo khoảng cách trong
    miền khởi tạo thực tế.
-   Chưa thấy thành phần reward tường minh thưởng cho mức tiến triển về
    phía mục tiêu. Position reward có thể tạo động lực giảm sai số,
    nhưng cần xác minh trong quá trình học và phân bố khởi tạo.
-   Reward yaw phạt sai lệch về 0. Nếu bài toán không yêu cầu yaw = 0,
    thành phần này có thể tạo ràng buộc không cần thiết. Nếu goal có yaw
    target, nên dùng sai số góc tương đối có xử lý wrap-around.
-   Điều kiện hover bonus không đề cập yaw. Đây có thể là chủ ý nếu yaw
    không quan trọng, nhưng cần nhất quán với yaw penalty.
-   Cần làm rõ `hover_bonus` được cộng mỗi bước hay theo sự kiện; nếu
    cộng mỗi bước, cần xác định cơ chế duy trì và tránh reward hacking.
-   Cần công bố công thức chính xác của các penalty, đơn vị, chuẩn hóa
    và cách scale theo `dt/reference_dt`. Việc scale reward theo dt
    không tự động bảo đảm mọi thành phần reward có ý nghĩa bất biến theo
    tần số.
-   `crash_penalty = 10.0` cần xác nhận dấu trong implementation là
    khoản trừ reward. Bảng chỉ ghi độ lớn dương, vì vậy không nên suy
    luận dấu nếu chưa xem code.

**Khuyến nghị:** vẽ đồ thị từng thành phần reward theo sai số vị trí,
tốc độ, góc nghiêng và motor signal. Từ đó xác định agent có động lực rõ
ràng để di chuyển về đích, giảm tốc và giữ hover hay không.

### 4.5. Termination và curriculum

Các điều kiện kết thúc sớm gồm collision, attitude vượt giới hạn, chạm
gần mặt đất, cách goal hơn 5 m và tốc độ lớn hơn 5 m/s. Curriculum đi
qua bảy stage từ attitude hover đến domain randomization.

**Điểm tích cực**

-   Có giới hạn an toàn để tránh trạng thái nguy hiểm kéo dài.
-   Curriculum từ nhiệm vụ đơn giản đến phức tạp là cách tiếp cận phù
    hợp với motor-level control.
-   Có điều kiện nâng stage dựa trên survival rate, position RMSE và
    attitude RMSE.
-   Metadata giúp phục hồi stage khi resume.

**Vấn đề cần kiểm tra**

1.  Cần công bố ngưỡng cụ thể của từng stage, số episode đánh giá, cửa
    sổ đánh giá và điều kiện đồng thời để nâng stage.
2.  Cần bảo đảm stage cuối có phân bố khởi tạo, disturbance và domain
    randomization đại diện cho điều kiện kiểm thử thực tế.
3.  Termination khi khoảng cách lớn hơn 5 m có thể loại bỏ các trạng
    thái mà policy cần học cách phục hồi. Cần chọn ngưỡng theo miền hoạt
    động dự kiến, không chỉ theo tiện lợi tính toán.
4.  Phải phân biệt `terminated` do thất bại vật lý với `truncated` do
    hết thời gian episode để xử lý bootstrap value đúng trong PPO.
5.  Cần xác nhận curriculum có thực sự bao gồm nhiệm vụ tiếp cận mục
    tiêu từ vị trí ban đầu khác nhau hay chủ yếu học hover tại chỗ. Tên
    stage chưa đủ để kết luận nội dung phân bố reset.

**Nhận định:** curriculum là điểm mạnh tiềm năng của nhánh motor, nhưng
phải kiểm tra cấu hình reset thực tế của từng stage. Nếu các stage không
tạo ra bài toán tiếp cận mục tiêu đủ đa dạng, policy có thể học hover
tốt nhưng không học được target reaching.

## 5. Những thiếu sót chung cần giải quyết

### 5.1. Chưa định nghĩa đầy đủ bài toán mục tiêu

Cần xác định rõ: - Goal là vị trí cố định hay thay đổi theo episode? -
Có yêu cầu yaw target hay chỉ cần giữ roll/pitch cân bằng? - Vùng chấp
nhận quanh goal có bán kính bao nhiêu? - Tốc độ tối đa khi tiếp cận và
tốc độ yêu cầu khi hover? - Thời gian duy trì liên tục để được tính
thành công? - Miền vị trí, vận tốc và attitude ban đầu?

Nếu không có các định nghĩa này, không thể xác định một cấu hình PPO có
đáp ứng bài toán hay không.

### 5.2. Thiếu chỉ số đánh giá tách biệt hai giai đoạn

Không nên chỉ sử dụng tổng reward để kết luận. Cần báo cáo ít nhất:

  -----------------------------------------------------------------------
  Nhóm                    Chỉ số đề xuất          Ý nghĩa
  ----------------------- ----------------------- -----------------------
  Tiếp cận                Success rate            Tỷ lệ episode đi vào
                                                  vùng mục tiêu

  Tiếp cận                Time-to-target          Thời gian đi vào vùng
                                                  mục tiêu

  Tiếp cận                Position RMSE           Sai số vị trí trong quá
                                                  trình tiếp cận

  Tiếp cận                Overshoot               Mức vượt quá mục tiêu

  Hover                   Hover success rate      Tỷ lệ duy trì đủ điều
                                                  kiện hover

  Hover                   Position RMSE           Độ dao động vị trí
                                                  quanh mục tiêu

  Hover                   Velocity RMSE           Độ dao động vận tốc

  Hover                   Roll/Pitch RMSE         Độ ổn định tư thế

  Hover                   Angular-rate RMSE       Độ ổn định vận tốc góc

  Điều khiển              Control effort          Mức sử dụng điều khiển

  Điều khiển              Saturation fraction     Tỷ lệ thời gian
                                                  actuator gần giới hạn

  Độ bền vững             Recovery rate           Khả năng phục hồi sau
                                                  nhiễu

  Độ bền vững             Crash rate              Tỷ lệ va chạm/thất bại
  -----------------------------------------------------------------------

Cần báo cáo kết quả trên nhiều seed và tập điều kiện đánh giá độc lập
với dữ liệu huấn luyện.

### 5.3. Thiếu kiểm thử nhiễu và sai lệch mô hình

Để đánh giá khả năng áp dụng, nên kiểm thử: - Khối lượng và inertia thay
đổi trong miền hợp lý. - Sai lệch operating point hoặc motor
effectiveness. - Nhiễu gió và gust. - Nhiễu cảm biến/ước lượng trạng
thái. - Vị trí, vận tốc và attitude ban đầu khác nhau. - Thay đổi goal
và độ cao mục tiêu nếu thuộc phạm vi ứng dụng.

Không nên đưa toàn bộ các biến thiên vào ngay từ đầu. Có thể tăng dần
theo curriculum, sau đó đánh giá trên các kịch bản chưa xuất hiện trong
huấn luyện.

## 6. Khuyến nghị cấu hình và quy trình thí nghiệm

### 6.1. Nếu chọn `velocity_ned` làm baseline

1.  Giữ nguyên state/action hiện tại trong thí nghiệm baseline.
2.  Xác minh action thực sự là velocity setpoint và giới hạn vận tốc
    theo từng trục.
3.  Tăng thời lượng episode từ 5 giây lên các mức thử nghiệm 10--20
    giây.
4.  Định nghĩa hover success là điều kiện đồng thời về vị trí, vận tốc
    và attitude được duy trì liên tục trong một khoảng thời gian.
5.  Kiểm tra lại phép quy đổi `MIN_ACTION_STD` và implementation của
    decay.
6.  Công bố công thức reward đầy đủ, nhất là `r_approach`, `r_pos` và
    `success bonus`.

### 6.2. Nếu chọn `motor_residual` làm phương án chính

1.  Xác minh hover operating point trước khi huấn luyện.
2.  Kiểm tra thứ tự actuator, mapping action-to-control-signal và RPM.
3.  Bổ sung hoặc xác minh observation sai số mục tiêu có hướng nếu goal
    thay đổi.
4.  Xác định rõ yaw target; bỏ hoặc sửa yaw penalty nếu yaw = 0 không
    phải yêu cầu.
5.  Xác minh curriculum có giai đoạn tiếp cận mục tiêu từ nhiều vị
    trí/độ cao/vận tốc ban đầu.
6.  Thử nghiệm gamma 0.995--0.999 cùng các chỉ số ổn định critic; không
    mặc định gamma cao hơn luôn tốt hơn.
7.  Theo dõi motor saturation, control effort, smoothness và khả năng
    giữ hover sau nhiễu.
8.  Đánh giá thời gian duy trì hover dài hơn thời lượng chỉ vừa đủ để
    nhận bonus.

### 6.3. Thiết kế đánh giá tối thiểu

Đề xuất tách thành ba bài kiểm thử:

**Test A -- Hover từ trạng thái gần cân bằng** - Khởi tạo drone gần mục
tiêu, vận tốc ban đầu nhỏ. - Đánh giá khả năng duy trì vị trí và
attitude. - Mục tiêu: xác nhận policy có khả năng hover cơ bản.

**Test B -- Target reaching** - Khởi tạo drone ở nhiều vị trí và vận tốc
ban đầu. - Đánh giá tỷ lệ tiếp cận vùng mục tiêu, thời gian tiếp cận và
overshoot. - Mục tiêu: xác nhận policy không chỉ học hover tại chỗ.

**Test C -- Reaching and sustained hovering** - Khởi tạo ở xa mục
tiêu. - Yêu cầu drone tiếp cận, giảm tốc và duy trì liên tục trong ít
nhất 5--10 giây. - Lặp lại với nhiều seed và disturbance. - Mục tiêu:
kiểm chứng đúng bài toán tổng thể.

Tiêu chí thành công phải được định nghĩa trước khi chạy test. Không điều
chỉnh ngưỡng sau khi đã xem kết quả chỉ để làm tăng success rate.

## 7. Thứ tự ưu tiên chỉnh sửa

  -----------------------------------------------------------------------
  Mức ưu tiên             Vấn đề                  Hành động
  ----------------------- ----------------------- -----------------------
  Rất cao                 Chưa định nghĩa đầy đủ  Chốt goal, vùng mục
                          target reaching và      tiêu, ngưỡng vận
                          sustained hovering      tốc/attitude và thời
                                                  gian duy trì

  Rất cao                 Chưa rõ công thức       Kiểm tra code, công bố
                          reward và điều kiện     công thức, xác minh
                          success                 thưởng không bị lặp sai

  Rất cao                 Episode `velocity_ned`  Thử horizon dài hơn và
                          chỉ 5 giây              đánh giá đủ thời gian
                                                  hover

  Cao                     Chưa chứng minh         Kiểm tra reset
                          curriculum motor có     distribution từng stage
                          target reaching         

  Cao                     Gamma motor 0.99 ở 50   Thử gamma cao hơn và
                          Hz                      theo dõi critic

  Cao                     Yaw penalty hướng về 0  Xác minh yaw target có
                                                  phải yêu cầu bài toán

  Cao                     Hover operating point   Kiểm thử độc lập trước
                          và motor mapping        PPO

  Trung bình              `K_EPOCHS = 30`         Theo dõi KL/clip
                                                  fraction; thử 10--20

  Trung bình              Action std decay        Xác minh công thức
                                                  decay và log std/action

  Trung bình              Thiếu đánh giá          Bổ sung disturbance và
                          robustness              domain randomization
                                                  evaluation
  -----------------------------------------------------------------------

## 8. Kết luận khoa học

Từ cấu hình được cung cấp, có thể nhận định:

1.  **`velocity_ned`:** có cấu trúc điều khiển và reward tương đối phù
    hợp với nhiệm vụ tiếp cận mục tiêu và hover ở mức vận tốc. Tuy
    nhiên, horizon 5 giây là hạn chế đáng kể đối với việc đánh giá đầy
    đủ quá trình tiếp cận, giảm tốc và duy trì cân bằng. Cần làm rõ
    reward và kiểm chứng thời gian hover dài hơn.
2.  **`motor_residual`:** có thiết kế phong phú hơn ở mức điều khiển
    actuator, bao gồm motor effort, smoothness, saturation và
    curriculum. Đây là hướng nghiên cứu có tiềm năng cho điều khiển trực
    tiếp quadcopter. Tuy nhiên, hiệu quả phụ thuộc lớn vào độ chính xác
    của hover calibration, motor mapping và việc curriculum có thực sự
    dạy policy tiếp cận mục tiêu hay không.
3.  **Về PPO:** các siêu tham số hiện tại là điểm khởi đầu hợp lý để
    thực nghiệm, nhưng chưa có cơ sở để gọi là tối ưu. `K_EPOCHS = 30`,
    gamma, learning rate và action standard deviation cần được đánh giá
    bằng log hội tụ và nhiều random seed.
4.  **Về mục tiêu nghiên cứu:** chưa thể kết luận policy đã giải quyết
    bài toán chỉ từ bảng cấu hình. Bằng chứng quyết định phải là kết quả
    kiểm thử độc lập cho cả target reaching và sustained hovering.

**Khuyến nghị cuối cùng:** trước hết hãy xác định một baseline có thể
lặp lại và kiểm thử riêng khả năng hover. Tiếp theo kiểm thử target
reaching từ nhiều điều kiện khởi tạo. Cuối cùng mới đánh giá bài toán
kết hợp tiếp cận và duy trì hover dưới nhiễu. Nếu mục tiêu nghiên cứu là
điều khiển trực tiếp bốn động cơ, nên tập trung phát triển và kiểm chứng
`motor_residual`; nếu mục tiêu là đánh giá PPO cho điều khiển quỹ
đạo/setpoint vận tốc, `velocity_ned` là baseline đơn giản và dễ phân
tích hơn.

------------------------------------------------------------------------

## 9. Thông tin cần có để đưa ra kết luận ở mức triển khai

Bảng cấu hình hiện tại chưa đủ để thẩm định toàn diện. Để đánh giá chính
xác hơn, cần đối chiếu thêm:

1.  Mã `get_reward()` của `velocity_ned` và `calculate_reward()` của
    `motor_residual`.
2.  Mã reset/termination/truncation của hai environment.
3.  Mã ánh xạ action sang velocity setpoint hoặc motor RPM.
4.  Cấu hình phân bố trạng thái khởi tạo và curriculum từng stage.
5.  Log training/evaluation gồm reward từng thành phần, success rate,
    position RMSE, attitude RMSE, velocity RMSE và crash rate.
6.  Một số quỹ đạo thử nghiệm thể hiện giai đoạn tiếp cận, giảm tốc và
    duy trì hover.

Khi có các dữ liệu này, có thể chuyển từ **đánh giá thiết kế cấu hình**
sang **đánh giá thực nghiệm policy**, xác định chính xác nút thắt nằm ở
reward design, state/action representation, PPO optimization hay mô hình
động lực học.
