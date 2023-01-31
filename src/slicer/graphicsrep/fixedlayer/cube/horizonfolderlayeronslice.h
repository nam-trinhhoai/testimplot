#ifndef HorizonFolderLayerOnSlice_H
#define HorizonFolderLayerOnSlice_H

#include "graphiclayer.h"
#include "sliceutils.h"
#include "curve.h"

class QGraphicsItem;
class HorizonFolderRepOnSlice;
class QGraphicsScene;
class QGLIsolineItem;
class IGeorefImage;
class CPUImagePaletteHolder;

class HorizonFolderLayerOnSlice : public GraphicLayer{
	  Q_OBJECT
public:
	  HorizonFolderLayerOnSlice(HorizonFolderRepOnSlice *rep,SliceDirection dir,
			int startValue,QGraphicsScene *scene,
			int defaultZDepth,QGraphicsItem *parent);
	virtual ~HorizonFolderLayerOnSlice();

	void setSliceIJPosition(int imageVal);

	virtual void show() override;
	virtual void hide() override;

	void internalShow();
	void internalHide();

    virtual QRectF boundingRect() const override;

   void setBuffer(CPUImagePaletteHolder* isoSurfaceHolder);

public slots:
	virtual void refresh() override;

protected:
	//QGLIsolineItem *m_lineItem;
	std::unique_ptr<Curve> m_curveMain;
	HorizonFolderRepOnSlice *m_rep;
	QTransform m_mainTransform;


	bool m_showInternal = false;
	bool m_showOK = false;

	CPUImagePaletteHolder* m_lastiso = nullptr;
};

#endif
