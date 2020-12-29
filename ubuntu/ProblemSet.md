## 题目1 时间同步与插值

1. 实现四元数NLerp，先线性插值，后模归一化。
```vim
struct Q{
    double x;
    double y;
    double z;
    double w;
};

Q NLerp(const Q& q1, const Q& q1, const double& t);
```


2. 实现函数，生成伪数据，包含相机拍照的时间戳cam_t，imu时间戳imu_t， imu的pose imu_q，

这里我们只用到IMU提供的姿态角，用四元数表示，忽略其他信息（加速度，角速度）。

假设：
> 相机为30Hz，IMU为100Hz，两个传感器均采集约5秒的数据
> 相机第一帧时间戳为0.0，imu第一个数据的时间戳为0.42。   
> 传感器第i帧的时间戳为 t_i = t_0 + i * (1 / freq) + 0.5 * (1 / freq) * rand()   
> IMU第一个数据为(0, 0, 0, 1), 第i个数据为 q_i = Normalize(q_(i-1) + 0.1 * (rand(), rand(), rand(), rand()))  
> rand返回值域为[0,1)


```vim
#define FREQ_CAM 30
#define FREQ_IMU 100
#define T0_CAM 0.0
#define T0_IMU 0.42
#define DURATION 5

void GenTimeStamp(std::vector<double>& cam_t, std::vector<double>& imu_t, std::vector<Q>& imu_q);

```

3. 针对每一个相机的时间戳tc，找到最近的两个imu时间戳，ti1, ti2，使得 ti1 <= tc <= ti2，
求tc时刻对应的imu的Pose（四元数表示），可能需要用线性插值NLerp。

```vim
bool CalImuPose(const double & tc, const std::vector<double>& imu_t, const std::vector<Q>& imu_q, Q qc);

```

4. 测试
```vim
int main(){
    //  do something



    //  find imu's pose for every cam_t
    Q qc;
    for(const auto& tc : cam_t){
        if( CalImuPose(tc, imu_t, imu_q, qc)){
            // print result 
        }
    }

    // return 0
}
```


## 题目2 生产者-消费者模型

生产者线程：数据采集，使用题目1中实现的功能，将 “相机时间戳+插值得到的四元数” 发送给消费者线程；  
消费者线程：使用数据(打印至屏幕输出即可)

注：1. 使用锁避免冲突
    2. (可选)使用CMAKE，并且以shared lib使用题目1中实现的功能


