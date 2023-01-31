#ifndef QVERTEX2D_H_
#define QVERTEX2D_H_

#include <QPoint>
#include <QVector2D>

class QVertex2D
{
public:
    QVertex2D();
    QVertex2D(const QVector2D &p, const QVector2D &c);
    QVector2D position; // position of the vertex
    QVector2D coords; // texture coordinates of the vertex
};

#endif /* QTLARGEIMAGEVIEWER_SRC_VIEW2D_VERTEX2D_H_ */
