#ifndef WellPickLayer3D_H
#define WellPickLayer3D_H

#include <QVector3D>
#include <QPhongMaterial>
#include "graphic3Dlayer.h"
#include "itooltipprovider.h"

class WellPickRep;
class WellPick;

namespace Qt3DCore {
	class QEntity;
	class QTransform;
}

namespace Qt3DRender {
	class QCamera;
}

class WellPickLayer3D : public Graphic3DLayer, public IToolTipProvider {
Q_OBJECT
public:
	WellPickLayer3D(WellPickRep*rep, QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera);
	virtual ~WellPickLayer3D();

	virtual void show() override;
	virtual void hide() override;

	virtual QRect3D boundingRect() const override;

	virtual void refresh() override;


	virtual void zScale(float val) override;

	void selectPick(int posX, int posY, QVector3D posGlobal);
	void deselectPick();

	virtual QString generateToolTipInfo() const override;

signals:
void showPickInfosSignal(const IToolTipProvider*,QString ,int,int, QVector3D);


public slots:
	void setDiameter(int value);
	void setThickness(int value);

private:
	WellPick * wellPick() const;

	WellPickRep* m_rep;
	double m_r = 50;
	float m_thickness= 15.0f;
/*	Qt3DCore::QEntity* m_entityL1;
	Qt3DCore::QTransform* m_transformL1;
	Qt3DCore::QEntity* m_entityL2;
	Qt3DCore::QTransform* m_transformL2;*/

	Qt3DCore::QEntity* m_disqueEntity;
	Qt3DCore::QTransform* m_transformDisque;
	Qt3DExtras::QPhongMaterial* material;
	bool m_isShown = false;
};

#endif
