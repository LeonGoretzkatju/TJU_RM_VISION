/*
 * 这是 TJU Robomasters 上位机源码，未经管理层允许严禁传播给其他人（包括队内以及队外）
 *
 * 该文件实现并封装了KM算法，用于计算最大权值匹配，适用于求解大神符格子与数字匹配、装甲灯柱匹配等问题
 */

#pragma once

#include<cstdio>
#include<memory.h>
#include<algorithm>// 使用其中的 min 函数
#include "util.hpp"
using namespace std;

const int SOLVER_MAX_SIZE = 500;

class KMSolver
{
protected:
    const float epsilon = 1e-7;
    float weight[SOLVER_MAX_SIZE][SOLVER_MAX_SIZE]; //X到Y的映射权重
    float lx[SOLVER_MAX_SIZE],ly[SOLVER_MAX_SIZE];//标号
    bool sx[SOLVER_MAX_SIZE],sy[SOLVER_MAX_SIZE];//是否被搜索过

    // 从 X(u) 寻找增广道路，找到则返回 true
    bool path(int u)
    {
        sx[u]=true;
        FOREACH(v,M)
            if(!sy[v] && lx[u]+ly[v]-weight[u][v] < epsilon)
            {
                sy[v]=true;
                if(match[v]==-1||path(match[v]))
                {
                    match[v]=u;
                    return true;
                }
            }
        return false;
    }

    float bestmatch()
    {
        // 初始化标号
        FOREACH(i,M)
        {
            lx[i]=-0x1FFFFFFF;
            ly[i]=0;
            FOREACH(j,M) if(lx[i]<weight[i][j])
                lx[i]=weight[i][j];
        }
        memset(match,-1,sizeof(match));
        FOREACH(u,M) while(1)
        {
            memset(sx,0,sizeof(sx));
            memset(sy,0,sizeof(sy));
            if(path(u)) break;
            // 修改标号
            float dx=999999999;
            FOREACH(i,M) if(sx[i]) FOREACH(j,M) if(!sy[j])
                dx=min(lx[i]+ly[j]-weight[i][j],dx);
            FOREACH(i,M)
            {
                if(sx[i]) lx[i]-=dx;
                if(sy[i]) ly[i]+=dx;
            }
        }
        float sum=0;
        FOREACH(i,M) sum=sum+weight[match[i]][i];
        return sum;
    }
public:
    int match[SOLVER_MAX_SIZE];//Y(i)与X(match [i])匹配
    int M;

    // 最大/最小 权匹配
    // matrix 为待匹配的权重方阵（按行展开为数组），匹配结果match[i]表示选择第i行第match[i]列的元素
    // m 方阵大小
    // isMax 是否要求最大化和
    // 返回 最大和的值
    int BestMatch(int matrix[],int m,bool isMax = true)
    {
        // 超过最大方阵尺寸
        assert(m <= SOLVER_MAX_SIZE);
        M = m;
        int opr = isMax ? 1 : -1;
        // 将原方阵转置，这是因为当前match[i]为选择第i列第match[i]行
        FORPAIR_ORDERED(i,j,m)
            weight[i][j] = opr * matrix[j*m+i];
        return (int)bestmatch() * opr;
    }

    // 最大/最小 浮点权匹配
    // matrix 为待匹配的权重方阵（按行展开为数组），匹配结果match[i]表示选择第i行第match[i]列的元素
    // m 方阵大小
    // isMax 是否要求最大化和
    // 返回最大和的值
    float BestMatch(float matrix[],int m,bool isMax = true)
    {
        // 超过最大方阵尺寸
        assert(m <= SOLVER_MAX_SIZE);
        M = m;
        float opr = isMax ? 1 : -1;
        // 由于方阵是浮点型，需要将浮点型映射到一个较大的整数范围去
        // 将原方阵转置，这是因为当前match[i]为选择第i列第match[i]行
        FORPAIR_ORDERED(i,j,m)
            weight[i][j] = opr * matrix[j*m+i];
        return bestmatch() * opr;
    }

    // 最大/最小 浮点权匹配
    // matrix 为待匹配的权重方阵，匹配结果match[i]表示选择第i行第match[i]列的元素
    // isMax 是否要求最大化和
    // 返回最大和的值
    float BestMatch(cv::Mat matrix,bool isMax = true)
    {
        assert(matrix.rows == matrix.cols && matrix.rows <= SOLVER_MAX_SIZE);
        M = matrix.rows;
        float opr = isMax ? 1 : -1;
        // 由于方阵是浮点型，需要将浮点型映射到一个较大的整数范围去
        // 将原方阵转置，这是因为当前match[i]为选择第i列第match[i]行
        FORPAIR_ORDERED(i,j,M)
            weight[i][j] = opr * matrix.at<float>(j,i);
        return bestmatch() * opr;
    }
};
