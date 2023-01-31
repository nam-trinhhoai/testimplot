/*
 * PointCtrl.h
 *
 *  Created on: Juin 2, 2022
 *      Author: l1049100 ( sylvain)
 */

#ifndef POINTCTRL_H_
#define POINTCTRL_H_

#include <QPointF>
#include "GraphEditor_GrabberItem.h"

class PointCtrl
{

public:
	QPointF m_position;
	QPointF m_ctrl1;
	QPointF m_ctrl2;
	int m_nbPoints;
	GraphEditor_GrabberItem *m_grab1;
	GraphEditor_GrabberItem *m_grab2;

	PointCtrl();

	PointCtrl(const PointCtrl& other);

	PointCtrl& operator=(const PointCtrl& other);


	PointCtrl(QPointF position, QPointF ctrl1);

	PointCtrl(QPointF position, QPointF ctrl1, QPointF ctrl2);


	~PointCtrl();


};



#endif
