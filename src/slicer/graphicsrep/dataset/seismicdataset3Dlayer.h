#ifndef SeismicDataset3DLayer_H
#define SeismicDataset3DLayer_H

#include "graphic3Dlayer.h"
#include <QMatrix4x4>

class DatasetRep;
class Seismic3DAbstractDataset;

namespace Qt3DCore {
	class QEntity;
	class QTransform;
}

class SeismicDataset3DLayer: public Graphic3DLayer {
Q_OBJECT
public:
	SeismicDataset3DLayer(DatasetRep *rep,QWindow * parent, Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~SeismicDataset3DLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void zScale(float val) override;

private:
	Seismic3DAbstractDataset * dataset() const;
protected slots:
	virtual void refresh() override;
protected:
	Qt3DCore::QEntity *m_volumeEntity;
	DatasetRep *m_rep;

	Qt3DCore::QTransform *m_transform;
	QMatrix4x4 m_transformMatrixOri;
};

#endif
