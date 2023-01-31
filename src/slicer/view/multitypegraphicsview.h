#ifndef MultiTypeGraphicsView_H
#define MultiTypeGraphicsView_H

#include "abstractgraphicsview.h"
//#include "randomlineview.h"
#include <QObject>
#include <QGraphicsItem>
#include <QMenu>
#include <QAction>
#include <QGraphicsSceneContextMenuEvent>

class WellBore;
class RandomLineView;
class AbstractInnerView;
//class eRandomType;

class MultiTypeGraphicsView: public AbstractGraphicsView {
Q_OBJECT
public:
	MultiTypeGraphicsView(WorkingSetManager *factory, QString uniqueName, QWidget *parent);
	virtual ~MultiTypeGraphicsView();

	RandomLineView* createRandomView(QPolygonF polygon,eRandomType eType = eTypeStandard);
	void unregisterRandomView(RandomLineView *pView) ;
	AbstractInnerView* createInnerView(ViewType viewType);
	void setInnerViewDefaultName(AbstractInnerView* innerView, QString newName);
protected slots:
	void addView(QAction * action);
	void openAddViewMenu();

private:
	void selectRandomActionWithUI();
	void addRandomFromWellBore(QList<WellBore*> well, double margin);

	QMenu* m_addViewMenu;
};

#endif
