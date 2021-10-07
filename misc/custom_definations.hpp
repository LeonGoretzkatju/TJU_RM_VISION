/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * 该文件定义了设备移植时 需要修改的代码部分，包括：
 * --选择机器人类型
 * --选择设备类型
 * --编辑程序路径
 * 对于每个设备，该文件一经移植就不需要再次修改了 更新程序不用更新该文件
 */

#pragma once


// 选择机器人类型（哨兵/步兵）
#define ROBOT_SENTINEL
// #define ROBOT_INFANCY
// #define ROBOT_HERO
// 选择设备（TX/秒算/PC）
//#define DEVICE_TX
#define DEVICE_MANIFOLD
//#define DEVICE_PC

// 程序文件路径
#define FILEDIR(fileName) "/home/dji/sentrytest/absoluteangle/tjurobomasteralgorithm/"#fileName

