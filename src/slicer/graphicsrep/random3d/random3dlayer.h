#ifndef Random3dLayer_H
#define Random3dLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "genericsurface3Dlayer.h"
#include "graymaterialinitializer.h"
//#include "surfacecollision.h"

class Random3dRep;
class RandomDataset;


class Random3dLayer: public Graphic3DLayer {
Q_OBJECT
public:
Random3dLayer(Random3dRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~Random3dLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;
	virtual void zScale(float val) override;

	//float distanceSigned(QVector3D position, bool* ok)override;

private:
	RandomDataset * randomData() const;

protected:


	Random3dRep *m_rep;


};

#endif
