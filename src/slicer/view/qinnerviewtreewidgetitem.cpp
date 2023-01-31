#include "qinnerviewtreewidgetitem.h"
#include "igraphicrepfactory.h"
#include "abstractgraphicrep.h"
#include "abstractinnerview.h"
#include "workingsetmanager.h"
#include "idata.h"
#include "monotypegraphicsview.h"
#include "qgraphicsreptreewidgetitem.h"

#include <QMenu>
#include <QAction>

#include "slicerep.h"

QInnerViewTreeWidgetItem::QInnerViewTreeWidgetItem(const QString &name,
		AbstractInnerView *view, WorkingSetManager *manager,
		QTreeWidgetItem *parent) :
		QTreeWidgetItem(parent), QObject(view) {
	m_view = view;
	if (parent!=nullptr && parent->treeWidget()!=nullptr && parent->treeWidget()->model()!=nullptr) {
		QSignalBlocker block(parent->treeWidget()->model());
		setData(0, Qt::DisplayRole, QVariant::fromValue(name));
		setData(0, Qt::UserRole, QVariant::fromValue(view));
		setToolTip(0,name);
	} else {
		setData(0, Qt::DisplayRole, QVariant::fromValue(name));
		setData(0, Qt::UserRole, QVariant::fromValue(view));
		setToolTip(0,name);
	}

	for (IData *data : manager->data())
		registerRepFactory(data->graphicRepFactory());

	connect(manager, SIGNAL(dataAdded(IData *)), this,
			SLOT(dataAdded(IData *)));
	connect(manager, SIGNAL(dataRemoved(IData *)), this,
			SLOT(dataRemoved(IData *)));

	setExpanded(false);
}

QTreeWidgetItem* QInnerViewTreeWidgetItem::findRepNode(AbstractGraphicRep *rep,
		QTreeWidgetItem *root) {
	for (int i = 0; i < root->childCount(); i++) {
		QTreeWidgetItem *c = root->child(i);
		QVariant var = c->data(0, Qt::UserRole);
		AbstractGraphicRep *test = var.value<AbstractGraphicRep*>();
		if (test != nullptr && test == rep)
			return c;

		QTreeWidgetItem *toFind = findRepNode(rep, c);
		if (toFind != nullptr)
			return toFind;
	}
	return nullptr;
}

QInnerViewTreeWidgetItem::~QInnerViewTreeWidgetItem() {

}
void QInnerViewTreeWidgetItem::registerRepFactory(IGraphicRepFactory *factory) {
	AbstractGraphicRep *rep = factory->rep(m_view->viewType(), m_view);
	if (rep == nullptr)
		return;

	QGraphicsRepTreeWidgetItem *item = new QGraphicsRepTreeWidgetItem(rep,
			factory, m_view, this);
	addChild(item);
}

void QInnerViewTreeWidgetItem::dataAdded(IData *d) {
	registerRepFactory(d->graphicRepFactory());
}

void QInnerViewTreeWidgetItem::dataRemoved(IData *d) {
	for (int i = 0; i < childCount(); i++) {
		QTreeWidgetItem *childItem = child(i);
		AbstractGraphicRep *rep = childItem->data(0, Qt::UserRole).value<
				AbstractGraphicRep*>();
		if (rep == nullptr)
			continue;
		if (rep->data() == d) {
			childItem->setCheckState(0, Qt::Unchecked);
			removeChild(childItem);
			delete childItem;
			break;
		}
	}
}

