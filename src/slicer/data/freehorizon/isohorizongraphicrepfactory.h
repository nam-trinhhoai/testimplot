#ifndef __ISOHORIZONGRAPHICREPFACTORY__
#define __ISOHORIZONGRAPHICREPFACTORY__
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class IsoHorizon;


class IsoHorizonGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	IsoHorizonGraphicRepFactory(IsoHorizon * data);
	virtual ~IsoHorizonGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;
	virtual QList<IGraphicRepFactory*> childReps(ViewType type, AbstractInnerView *parent) override;

	/*
private slots:
	void wellPickAdded(WellPick* wellPick);
	void wellPickRemoved(WellPick* wellPick);
	*/

private:
	IsoHorizon * m_data;

};

#endif
