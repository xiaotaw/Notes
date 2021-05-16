/**
 * @file rosbag_reader.hpp
 * @author xiaotaw (you@domain.com)
 * @brief RosbagReader
 * @version 0.1
 * @date 2021-01-07
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <eigen_conversions/eigen_msg.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>

#include <algorithm>  // std::transform
#include <iterator>   // std::back_inserter
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @brief RosbagReader class
 *
 */
class RosbagReader {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  /**
   * @brief Construct a new Rosbag Reader object
   *
   * @param bag_path [IN] the path of rosbag
   */
  RosbagReader(const std::string& bag_path) : bag_path_(bag_path) {
  }

  /**
   * @brief read PoseStamped msgs
   *
   * @param topic_name [IN] the topic tobe read
   * @param msgs [OUT] a vector of msgs has been read
   * @return int, the number of msgs read
   */
  int readPoseStamped(const std::string& topic_name, std::vector<geometry_msgs::PoseStamped>& msgs) {
    return readTopic<geometry_msgs::PoseStamped>(topic_name, msgs);
  }

  int readPoseStamped(const std::string& topic_name, std::vector<Eigen::Matrix4d>& poses) {
    std::vector<geometry_msgs::PoseStamped> msgs;
    int res = readTopic<geometry_msgs::PoseStamped>(topic_name, msgs);

    for (const geometry_msgs::PoseStamped& msg : msgs) {
      // convert pose msgs to eigen matrix
      Eigen::Affine3d pose;
      tf::poseMsgToEigen(msg.pose, pose);
      poses.emplace_back(pose.matrix());
    }
    return res;
  }

  /**
   * @brief read pointcloud2 msgs
   *
   * @param topic_name [IN] the topic tobe read
   * @param msgs [OUT] a vector of msgs has been read
   * @return int, the number of msgs read
   */
  int readPointCloud2(const std::string& topic_name, std::vector<sensor_msgs::PointCloud2>& msgs) {
    return readTopic<sensor_msgs::PointCloud2>(topic_name, msgs);
  }

  int readPointCloud2(const std::string& topic_name, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) {
    std::vector<sensor_msgs::PointCloud2> msgs;
    int res = readTopic<sensor_msgs::PointCloud2>(topic_name, msgs);

    for (const sensor_msgs::PointCloud2& msg : msgs) {
      // convert pc2 msgs to pcl
      auto pcl_pc2_tmp = boost::make_shared<pcl::PCLPointCloud2>();
      auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      pcl_conversions::toPCL(msg, *pcl_pc2_tmp);
      pcl::fromPCLPointCloud2(*pcl_pc2_tmp, *cloud);
      
      clouds.emplace_back(cloud);
    }
    return res;
  }

  /**
   * @brief template func of reading messages by topic name
   *
   * @tparam MsgType, the message type
   * @param topic_name [IN] the topic tobe read
   * @param msgs [OUT] a vector of msgs has been read
   * @return int, the number of msgs has been read
   */
  template <class MsgType>
  int readTopic(const std::string& topic_name, std::vector<MsgType>& msgs) {
    rosbag::Bag bag;
    bag.open(bag_path_, rosbag::bagmode::Read);
    std::vector<std::string> topic_names = {topic_name};
    rosbag::View view(bag, rosbag::TopicQuery(topic_names));

    msgs.clear();

    for (rosbag::MessageInstance const msg : view) {
      typename MsgType::ConstPtr msg_ptr = msg.instantiate<MsgType>();
      if (msg_ptr != nullptr) {
        msgs.emplace_back(*msg_ptr);
      } else {
        std::cerr << "[Error]: " << __FILE__ << "Instantiate msg type error" << std::endl;
      }
    }
    ROS_INFO_STREAM("[INFO]: Read " << msgs.size() << " messages");
    bag.close();
    return msgs.size();
  }

  enum kMsgs
  {
    kPoseStamped = 0,
    kPointCloud2 = 1,
    kImu = 2
  };

  using TopicNameType = std::unordered_map<std::string, kMsgs>;
  using TopicMsgs = std::unordered_map<std::string, std::vector<void*>>;

  /**
   * @brief Read many topic at one time (not finished yet)
   *
   * @param topic_name_type [IN]
   * @param topic_msgs [OUT]
   * @return int
   */
  int readTopic(const TopicNameType topic_name_type, TopicMsgs& topic_msgs) {
    std::cerr << "[FATAL]: Not Implemented yet!" << std::endl;
    return 0;

    rosbag::Bag bag;
    bag.open(bag_path_, rosbag::bagmode::Read);

    // get Map's Keys into a Vector
    std::vector<std::string> topic_names;
    std::transform(
      topic_name_type.begin(), topic_name_type.end(), std::back_inserter(topic_names),
      [](const std::pair<std::string, kMsgs>& kv) { return kv.first; } /* need c++14 for 'auto' in lambda */
    );

    rosbag::View view(bag, rosbag::TopicQuery(topic_names));

    for (rosbag::MessageInstance const msg : view) {
      std::string topic_name = msg.getTopic();
    }
    return 0;
  }

private:
  std::string bag_path_;
};
