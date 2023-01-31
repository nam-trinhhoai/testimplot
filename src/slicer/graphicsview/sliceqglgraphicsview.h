#ifndef SliceQGLGraphicsView_H
#define SliceQGLGraphicsView_H

#include "baseqglgraphicsview.h"

class SliceQGLGraphicsView : public BaseQGLGraphicsView
{
   Q_OBJECT
public:
   explicit SliceQGLGraphicsView(QWidget *parent = 0);

protected:
  void wheelEvent(QWheelEvent * e)override;
};

#endif
