/*
 * vertex2D.cpp
 *
 *  Created on: 18 juin 2018
 *      Author: j0334308
 */

#include "qvertex2D.h"

QVertex2D::QVertex2D(){}
QVertex2D::QVertex2D(const QVector2D &p, const QVector2D &c) :
        position(p)
      , coords(c)
{
}
