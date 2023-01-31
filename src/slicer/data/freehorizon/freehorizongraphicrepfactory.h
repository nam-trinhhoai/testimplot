#ifndef __FREEHORIZONGRAPHICREPFACTORY__
#define __FREEHORIZONGRAPHICREPFACTORY__
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
#include <freehorizon.h>

class FreeHorizonGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	FreeHorizonGraphicRepFactory(FreeHorizon * data);
	virtual ~FreeHorizonGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;
	virtual QList<IGraphicRepFactory*> childReps(ViewType type, AbstractInnerView *parent) override;

	/*
private slots:
	void wellPickAdded(WellPick* wellPick);
	void wellPickRemoved(WellPick* wellPick);
	*/

private:
	FreeHorizon * m_data;

private slots:
	void attributAdded(FreeHorizon::Attribut *data);
	void attributRemoved(FreeHorizon::Attribut *data);

};

#endif
