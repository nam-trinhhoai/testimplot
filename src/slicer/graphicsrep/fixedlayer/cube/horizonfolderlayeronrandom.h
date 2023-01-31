#ifndef HorizonFolderLayerOnRandom_H
#define HorizonFolderLayerOnRandom_H

#include "graphiclayer.h"
#include "sliceutils.h"
#include "curve.h"

class QGraphicsItem;
class HorizonFolderRepOnRandom;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;
class CPUImagePaletteHolder;
class Affine2DTransformation;

class HorizonFolderLayerOnRandom : public GraphicLayer{
	  Q_OBJECT
public:
	  HorizonFolderLayerOnRandom(HorizonFolderRepOnRandom *rep,
			int startValue,QGraphicsScene *scene,
			int defaultZDepth,QGraphicsItem *parent);
	virtual ~HorizonFolderLayerOnRandom();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

	void internalShow();
	void internalHide();

    virtual QRectF boundingRect() const override;

   void setBuffer(CPUImagePaletteHolder* isoSurfaceHolder);

private :
   void computeTransform();

public slots:
	virtual void refresh() override;

protected:
	//QGLIsolineItem *m_lineItem;
	//std::unique_ptr<Curve> m_curveMain;
	HorizonFolderRepOnRandom *m_rep;
	QTransform m_mainTransform;

	Affine2DTransformation* m_transformation=nullptr;


	bool m_showInternal = false;
	bool m_showOK = false;

	CPUImagePaletteHolder* m_lastiso = nullptr;

	QPolygon m_discreatPolyline;
};

#endif
