#!/usr/bin/python
# modified on: https://gist.github.com/guilhermelawless/9e5b8ebaae6fbdc8fd3f6711f7ad8106
# 

'''
Description: Concatenates two ROS bags, adjusting time and transformation offsets.
Useful for the purpose of evaluating robot kidnapping events in pre-recorded datasets.

Assumptions:
    -Only one TF should be in the bags: the base_link -> laser_frame TF (script can be improved to allow for more)

Extra:
    -Install progressbar (available in pip) to see progress during the script
'''

import rospy
import rosbag
import sys
from genpy.rostime import Time as gTime
from tf2_kdl import transform_to_kdl
from tf_conversions import toMsg
from geometry_msgs.msg import TransformStamped

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('Usage: {} BAG_1 BAG_2 OUTPUT_BAG'.format(sys.argv[0]))
        sys.exit()

    bag1_wrap = rosbag.Bag(sys.argv[1],'r')
    bag1_end = bag1_wrap.get_end_time()

    bag2_wrap = rosbag.Bag(sys.argv[2],'r')
    bag2_start = bag2_wrap.get_start_time()

    out_wrap = rosbag.Bag(sys.argv[3],'w')

    max_value = bag1_wrap.get_message_count()
    print(max_value)

    try:
        from progressbar import ProgressBar
        bar = ProgressBar(maxval = max_value)
        bar.start()
    except:
        print('Working...')
        bar = None

    save_tf = None
    i = 0
    for topic, msg, time in bag1_wrap.read_messages():
        if topic == "/tf" and msg.transforms:
            save_tf = msg
        out_wrap.write(topic, msg, time)
        if bar: bar.update(i)
        i = i + 1
    if bar: bar.finish()

    if bag2_start >= bag1_end:
        offset = gTime.from_sec(bag2_start - bag1_end)
        offset_flag = 1
    else:
        offset = gTime.from_sec(bag1_end - bag2_start)
        offset_flag = -1

    if save_tf:
        save_ts = save_tf.transforms[0]
        save_kdl = transform_to_kdl(save_ts)

    ts = TransformStamped()
    if bar: 
        bar = ProgressBar(maxval = bag2_wrap.get_message_count())
        bar.start()
    i = 0
    for topic, msg, time in bag2_wrap.read_messages():
        # time is genpy.Time
        if offset_flag > 0:
            new_time = gTime.from_sec( time.to_sec() - offset.to_sec() )
        else:
            new_time = gTime.from_sec( time.to_sec() + offset.to_sec() )
        if topic == "/tf" and msg.transforms and save_tf:
            ts = msg.transforms[0]
            pose = toMsg( save_kdl * transform_to_kdl(ts) )
            if offset_flag > 0:
                msg.transforms[0].header.stamp -= rospy.Duration(offset.to_sec())
            else:
                msg.transforms[0].header.stamp += rospy.Duration(offset.to_sec())
            msg.transforms[0].transform.translation = pose.position
            msg.transforms[0].transform.rotation = pose.orientation
        else:
            if msg._has_header:
                if offset_flag > 0:
                    msg.header.stamp -= rospy.Duration(offset.to_sec())
                else:
                    msg.header.stamp += rospy.Duration(offset.to_sec())

        out_wrap.write(topic, msg, new_time)
        if bar: bar.update(i)
        i = i + 1
    if bar: bar.finish()
    out_wrap.close()