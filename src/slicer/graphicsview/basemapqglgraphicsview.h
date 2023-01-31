#ifndef BaseMapQGLGraphicsView_H_
#define BaseMapQGLGraphicsView_H_

#include "baseqglgraphicsview.h"

class BaseMapQGLGraphicsView : public BaseQGLGraphicsView
{
      Q_OBJECT
public:
   BaseMapQGLGraphicsView(QWidget *parent = 0);

   virtual std::pair<float, float> resetZoom(void) override;
   virtual std::pair<float, float> setVisibleRect(const QRectF &bbox) override;
protected:
   void wheelEvent(QWheelEvent * e)override;

   bool first = true;
};

#endif
