#ifndef QInnerViewTreeWidgetItem_H
#define QInnerViewTreeWidgetItem_H

#include <QTreeWidgetItem>

class AbstractGraphicRep;
class IGraphicRepFactory;
class AbstractInnerView;
class WorkingSetManager;
class IData;

//I'm a node representing the content of a viewer

//Shortcut: this double inheriting is not really good as QObject mecanism are expensive.
//In case of too many nodes, A custom model should be implemented
class QInnerViewTreeWidgetItem:  public QObject ,public QTreeWidgetItem{
	Q_OBJECT
public:
	QInnerViewTreeWidgetItem(const QString & name,AbstractInnerView * view,WorkingSetManager * manager,QTreeWidgetItem *parent);
	virtual ~QInnerViewTreeWidgetItem();

	AbstractInnerView * innerView() const{return m_view;}

	static QTreeWidgetItem* findRepNode(AbstractGraphicRep *rep,QTreeWidgetItem * root);

private slots:
	void dataAdded(IData *d);
	void dataRemoved(IData *d);

private:
	void registerRepFactory(IGraphicRepFactory *factory) ;

private:
	AbstractInnerView * m_view;
	WorkingSetManager *m_manager;
};

#endif
