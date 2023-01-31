#ifndef MarkerGraphicRepFactory_H
#define MarkerGraphicRepFactory_H
#include <QObject>
#include "viewutils.h"

#include "igraphicrepfactory.h"
class Marker;
class WellPick;

class MarkerGraphicRepFactory :public IGraphicRepFactory{
	Q_OBJECT
public:
	MarkerGraphicRepFactory(Marker * data);
	virtual ~MarkerGraphicRepFactory();

	virtual AbstractGraphicRep * rep(ViewType type,AbstractInnerView * parent)  override;
	virtual QList<IGraphicRepFactory*> childReps(ViewType type,
			AbstractInnerView *parent) override;

private slots:
	void wellPickAdded(WellPick* wellPick);
	void wellPickRemoved(WellPick* wellPick);

private:
	Marker * m_data;

};

#endif
