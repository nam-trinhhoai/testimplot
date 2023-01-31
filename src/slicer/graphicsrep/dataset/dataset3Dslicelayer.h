#ifndef Dataset3DSliceLayer_H
#define Dataset3DSliceLayer_H

#include <QVector3D>
#include <QMatrix4x4>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "qt3dhelpers.h"
#include <QMouseEvent>

class Dataset3DSliceRep;
class Seismic3DAbstractDataset;
class ColorTableTexture;
class CudaImageTexture;
class CUDAImagePaletteHolder;

namespace Qt3DCore {
	class QEntity;
	class QTransform;
}

namespace Qt3DRender {
	class QMaterial;
	class QParameter;
	class QPickEvent;
}
namespace Qt3DInput
{
	class QMouseEvent;
}

class Dataset3DSliceLayer : public Graphic3DLayer {
Q_OBJECT
public:
	Dataset3DSliceLayer(Dataset3DSliceRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~Dataset3DSliceLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;


	virtual void zScale(float val) override;

	  signals:
		void sendAnimationCam(int button,QVector3D pos);

private:
	Seismic3DAbstractDataset* dataset() const;
	void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );

	void selectPlane(bool visible);
	void movePlane(int decal);
	void movePlane(QVector3D decal);

protected slots:
	void update();
	void updateLookupTable(const LookupTable &table);

	void opacityChanged(float val);
	void rangeChanged(const QVector2D & value);

protected:
	Qt3DCore::QEntity *m_sliceEntity;
	ColorTableTexture * m_colorTexture;
	CudaImageTexture * m_cudaTexture;

	Qt3DRender::QMaterial *m_material;
	Qt3DRender::QParameter * m_opacityParameter;
	Qt3DRender::QParameter * m_hoverParameter;

	Qt3DRender::QParameter * m_paletteRangeParameter;

	Qt3DCore::QTransform *m_transform;
	QVector3D m_sliceVector;

	Dataset3DSliceRep *m_rep;

	QMatrix4x4 m_matrixTr;
	QMatrix4x4 m_scaleTr;
	QMatrix4x4 m_rotationTr;
	QMatrix4x4 m_transTr;

	Qt3DCore::QEntity* m_line1;
	Qt3DCore::QEntity* m_line2;
	Qt3DCore::QEntity* m_line3;
	Qt3DCore::QEntity* m_line4;

	bool m_movable2=false;
	bool m_movable=false;
	QPointF m_lastPos;
	int m_current=0;
	QVector3D m_lastPosWorld;

	Qt3DRender::QObjectPicker *picker;
	Qt3DRender::QObjectPicker *picker2;
	 bool m_actifRayon = false;
};

#endif
