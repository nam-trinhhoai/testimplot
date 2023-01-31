#ifndef HorizonFolderLayerOnMap_H
#define HorizonFolderLayerOnMap_H

#include "graphiclayer.h"
class QGraphicsItem;
class RGBInterleavedQGLCUDAImageItem;
//class FixedRGBLayersFromDatasetAndCubeRep;
class HorizonDataRep;
class QGraphicsScene;
class QGLImageFilledHistogramItem;
class CUDARGBInterleavedImage;
class CPUImagePaletteHolder;

class HorizonFolderLayerOnMap : public GraphicLayer{
	  Q_OBJECT
public:
	  HorizonFolderLayerOnMap(HorizonDataRep *rep,QGraphicsScene *scene,
			 int defaultZDepth,QGraphicsItem *parent);
	virtual ~HorizonFolderLayerOnMap();

	virtual void show() override;
	virtual void hide() override;

    virtual QRectF boundingRect() const override;

    void setBuffer(CUDARGBInterleavedImage* image,CPUImagePaletteHolder* isoSurfaceHolder);

    void internalShow();
    void internalHide();

public slots:
	virtual void refresh() override;
	void updateRgb();
	void updateIsoSurface();
	void rangeChanged(unsigned int i, const QVector2D v);

protected slots:
	void minValueActivated(bool activated);
	void minValueChanged(float value);

protected:
	RGBInterleavedQGLCUDAImageItem *m_item;

	HorizonDataRep *m_rep;

private:
	CUDARGBInterleavedImage* m_lastimage=nullptr;
	CPUImagePaletteHolder* m_lastiso = nullptr;

	bool m_showOK = false;
	bool m_showInternal = false;

	QGraphicsItem *m_parent;
};

#endif
