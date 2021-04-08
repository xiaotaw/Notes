/**
 * @file cam_intr.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2021-04-11
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

struct CamIntr {
  double fx, fy, cx, cy;

  CamIntr() {}

  CamIntr(double _fx, double _fy, double _cx, double _cy)
      : fx(_fx), fy(_fy), cx(_cx), cy(_cy) {}
};

struct CamIntrInv {
  double fx_inv, fy_inv, cx, cy;

  CamIntrInv() {}

  CamIntrInv(double _fx_inv, double _fy_inv, double _cx, double _cy)
      : fx_inv(_fx_inv), fy_inv(_fy_inv), cx(_cx), cy(_cy) {}

  CamIntrInv(const CamIntr &cam_intr)
      : fx_inv(1.0 / cam_intr.fx), fy_inv(1.0 / cam_intr.fy), cx(cam_intr.cx),
        cy(cam_intr.cy) {}
};
