#ifndef QGraphicsRepTreeWidgetItem_H
#define QGraphicsRepTreeWidgetItem_H

#include "viewutils.h"

#include <QPointer>
#include <QTreeWidgetItem>

class AbstractGraphicRep;
class AbstractInnerView;
class IGraphicRepFactory;
class ITreeWidgetItemDecorator;

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class QGraphicsRepTreeWidgetItem:  public QObject ,public QTreeWidgetItem{
	Q_OBJECT

public:
	QGraphicsRepTreeWidgetItem(AbstractGraphicRep *rep,IGraphicRepFactory *factory,AbstractInnerView * view,QTreeWidgetItem *parent);
	virtual ~QGraphicsRepTreeWidgetItem();
	QTreeWidgetItem* addChildRep(AbstractGraphicRep *rep);

	const AbstractGraphicRep* getRep() const;
	AbstractGraphicRep* getRep();

private slots:
	//Dynamic rep insertion
	void insertChildRep(AbstractGraphicRep * rep);
	//Data changed
	void childAdded(IGraphicRepFactory *child);
	void childRemoved(IGraphicRepFactory *child);

	void nameChanged();

	void dataDisplayPreferenceChanged(std::vector<ViewType>, bool);
	void deletedRep(AbstractGraphicRep * rep);

	void updateItemWithDecorator();
signals:
   void repDeleted(AbstractGraphicRep *);

private:
   void connectChildRep(AbstractGraphicRep *childRep);
	QGraphicsRepTreeWidgetItem * generateChild(IGraphicRepFactory *child);
	void setRep(AbstractGraphicRep * rep); // MZR Test
private:
	AbstractInnerView * m_view;
	AbstractGraphicRep *m_rep;
	QPointer<ITreeWidgetItemDecorator> m_decorator;
};

#endif
