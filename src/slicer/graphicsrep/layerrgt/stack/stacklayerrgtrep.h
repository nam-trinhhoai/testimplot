#ifndef StackLayerRGTRep_H
#define StackLayerRGTRep_H

#include <QObject>
#include "abstractgraphicrep.h"
#include "imouseimagedataprovider.h"
#include "isliceablerep.h"
#include "iGraphicToolDataControl.h"
#include "iCUDAImageClone.h"

class CUDAImagePaletteHolder;
class QGLLineItem;
class StackLayerRGTPropPanel;
class StackLayerRGTLayer;
class LayerSlice;

//For BaseMap
class StackLayerRGTRep: public AbstractGraphicRep, public IMouseImageDataProvider,
public ISliceableRep, public iGraphicToolDataControl, public iCUDAImageClone {
Q_OBJECT
public:
	StackLayerRGTRep(LayerSlice * layerSlice, AbstractInnerView *parent = 0);
	virtual ~StackLayerRGTRep();

	LayerSlice* layerSlice() const;

	CUDAImagePaletteHolder* isoSurfaceHolder();
	CUDAImagePaletteHolder* image() {
		return m_image;
	}

	QString getLabelFromPosition(int val);
	QString getCurrentLabel();

	void showCrossHair(bool val);
	bool crossHair() const;

	//AbstractGraphicRep
	QWidget* propertyPanel() override;
	GraphicLayer * layer(QGraphicsScene *scene, int defaultZDepth,QGraphicsItem *parent)override;
	Graphic3DLayer * layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera) override;

	IData* data() const override;

	//IMouseImageDataProvider
	virtual bool mouseData(double x, double y,MouseInfo & info) override;
	virtual void buildContextMenu(QMenu * menu) override;
	//ISliceableRep
	virtual void setSliceIJPosition(int val) override;
	int currentSliceIJPosition() const;

	// iGraphicToolDataControl
	void deleteGraphicItemDataContent(QGraphicsItem *item) override;

	QVector2D stackRange() const;
	virtual TypeRep getTypeGraphicRep() override;

	QGraphicsObject* cloneCUDAImageWithMask(QGraphicsItem *parent) override;

private slots:
	void dataChanged();
	void updateNbOutputSlices(int nbOutputSlices);
	void deleteStackLayerRGTRep();
signals:
	void sliceIJPositionChanged(int pos);
	void stackRangeChanged(QVector2D);
	void deletedRep(AbstractGraphicRep *rep);// MZR 15072021
private:
	StackLayerRGTPropPanel *m_propPanel;
	StackLayerRGTLayer *m_layer;

	LayerSlice * m_layerSlice;

	CUDAImagePaletteHolder *m_image;

	bool m_showCrossHair;
	int m_currentStackIndex;
};

#endif
