/**
 * camera params
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/18 08:40
 */
#pragma once
class CameraParams
{
public:
    unsigned rows_;
    unsigned cols_;
    double fx_;
    double fy_;
    double cx_;
    double cy_;

    double k1_;
    double k2_;
    double p1_;
    double p2_;
    double k3_;

    CameraParams(){}

    CameraParams(unsigned rows, unsigned cols, double fx, double fy, double cx,
                 double cy) : rows_(rows), cols_(cols), fx_(fx), fy_(fy),
                              cx_(cx), cy_(cy) {}
                              
    CameraParams(unsigned rows, unsigned cols, double fx, double fy, double cx,
                 double cy, double k1, double k2, double p1, double p2,
                 double k3) : rows_(rows), cols_(cols), fx_(fx), fy_(fy),
                              cx_(cx), cy_(cy), k1_(k1), k2_(k2), p1_(p1),
                              p2_(p2), k3_(k3) {}
};