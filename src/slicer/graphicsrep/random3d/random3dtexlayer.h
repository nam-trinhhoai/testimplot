#ifndef Random3dTexLayer_H
#define Random3dTexLayer_H

#include <QVector3D>
#include "graphic3Dlayer.h"
#include "lookuptable.h"
#include "genericsurface3Dlayer.h"
#include "graymaterialinitializer.h"
//#include "surfacecollision.h"

class Random3dTexRep;
class RandomTexDataset;


class Random3dTexLayer: public Graphic3DLayer {
Q_OBJECT
public:
Random3dTexLayer(Random3dTexRep *rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~Random3dTexLayer();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;
	virtual void zScale(float val) override;


private:
	RandomTexDataset * randomData() const;

protected:


	Random3dTexRep *m_rep;


};

#endif
