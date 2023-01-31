#include "qgraphicsreptreewidgetitem.h"
#include "qinnerviewtreewidgetitem.h"
#include "igraphicrepfactory.h"
#include "abstractgraphicrep.h"
#include "abstractinnerview.h"
#include "layerdatasetgrahicrepfactory.h"
#include "idata.h"
#include <QDebug>
#include <QColor>
#include <QFont>
#include <QBrush>

#include "slicerep.h"
#include "datasetrep.h"
#include "stacklayerrgtrep.h"
#include "layerrgtrep.h"
#include "wellpickreponslice.h"
#include "rgblayerrgtrep.h"
#include "layerrgtreponslice.h"
#include "dataset3Dslicerep.h"
#include "wellborereponslice.h"
#include "wellborerepon3d.h"
#include "wellborereponmap.h"
#include "randomrep.h"
#include "wellborereponrandom.h"
#include "wellpickreponrandom.h"
#include "itreewidgetitemdecorator.h"
#include "itreewidgetitemdecoratorprovider.h"

QGraphicsRepTreeWidgetItem::QGraphicsRepTreeWidgetItem(AbstractGraphicRep *rep,
		IGraphicRepFactory *factory, AbstractInnerView *view,
		QTreeWidgetItem *parent) :
		QTreeWidgetItem(parent), QObject(view) {
	m_view = view;
	m_rep = rep;

	if (parent!=nullptr && parent->treeWidget()!=nullptr && parent->treeWidget()->model()!=nullptr) {
		QSignalBlocker block(parent->treeWidget()->model());
		setData(0, Qt::DisplayRole, QVariant::fromValue(rep->name()));
		setToolTip(0,rep->name());

		setData(0, Qt::UserRole, QVariant::fromValue(rep));
		if (rep->canBeDisplayed())
			setData(0, Qt::CheckStateRole, QVariant::fromValue(false));
	} else {
		setData(0, Qt::DisplayRole, QVariant::fromValue(rep->name()));
		setToolTip(0,rep->name());
		setData(0, Qt::UserRole, QVariant::fromValue(rep));
		if (rep->canBeDisplayed())
			setData(0, Qt::CheckStateRole, QVariant::fromValue(false));
	}
	connect(rep, SIGNAL(insertChildRep(AbstractGraphicRep *)), this,
			SLOT(insertChildRep(AbstractGraphicRep *)));

	connect(rep, SIGNAL(nameChanged()), this, SLOT(nameChanged()));

	if (factory != nullptr) {
		QList<IGraphicRepFactory*> reps = factory->childReps(m_view->viewType(),
				m_view);
		QList<QTreeWidgetItem*> itemsToAdd;
		for (IGraphicRepFactory *ff : reps) {
			QGraphicsRepTreeWidgetItem *el = generateChild(ff);
			if (el != nullptr)
			{
				itemsToAdd.push_back(el);
			}
		}
		addChildren(itemsToAdd);
		connect(factory, SIGNAL(childAdded(IGraphicRepFactory * )), this,
				SLOT(childAdded(IGraphicRepFactory*)));
		connect(factory, SIGNAL(childRemoved(IGraphicRepFactory * )), this,
				SLOT(childRemoved(IGraphicRepFactory*)));
	}
	setExpanded(false);

	if (rep->canBeDisplayed()) {
		connect(rep->data(), &IData::displayPreferenceChanged, this, &QGraphicsRepTreeWidgetItem::dataDisplayPreferenceChanged);
	}

	if (ITreeWidgetItemDecoratorProvider* provider = dynamic_cast<ITreeWidgetItemDecoratorProvider*>(m_rep->data())) {
		m_decorator = provider->getTreeWidgetItemDecorator();
		m_decorator->decorate(this);
		connect(m_decorator.data(), &ITreeWidgetItemDecorator::decoratorUdpated, this, &QGraphicsRepTreeWidgetItem::updateItemWithDecorator);
	}
}

void QGraphicsRepTreeWidgetItem::dataDisplayPreferenceChanged(std::vector<ViewType> viewTypesChanged,
		bool preferenceChanged) {
	auto it = std::find(viewTypesChanged.begin(), viewTypesChanged.end(), m_view->viewType());
	bool update = it!=viewTypesChanged.end();

	if (update) {
		bool newPreference = m_rep->data()->displayPreference(m_view->viewType());
		if (this->data(0, Qt::CheckStateRole).toBool()!=newPreference) {
			Qt::CheckState state = newPreference ? Qt::Checked : Qt::Unchecked;
			this->setCheckState(0, state);
		}
	}
}

void QGraphicsRepTreeWidgetItem::nameChanged() {
	setData(0, Qt::DisplayRole, QVariant::fromValue(m_rep->name()));
}
void QGraphicsRepTreeWidgetItem::insertChildRep(AbstractGraphicRep *rep) {
	QGraphicsRepTreeWidgetItem *childNode = new QGraphicsRepTreeWidgetItem(rep,
			nullptr, m_view, this);


	connectChildRep(rep);
	addChild(childNode);

	Dataset3DSliceRep* repdataset  = dynamic_cast<Dataset3DSliceRep*>(rep);
	if( repdataset != nullptr)
	{
		qDebug()<<" checked";
		childNode->setData(0,Qt::CheckStateRole,Qt::Checked );
	}
}

QGraphicsRepTreeWidgetItem::~QGraphicsRepTreeWidgetItem() {
	if (!m_decorator.isNull()) {
		disconnect(m_decorator.data(), &ITreeWidgetItemDecorator::decoratorUdpated, this, &QGraphicsRepTreeWidgetItem::updateItemWithDecorator);
	}
	if(m_rep != nullptr){
		delete m_rep;
	}
}

void QGraphicsRepTreeWidgetItem::childAdded(IGraphicRepFactory *child) {
	QGraphicsRepTreeWidgetItem *childNode = generateChild(child);
	if (child == nullptr)
		return;
	// qDebug() << childNode->getRep()->name();
	addChild(childNode);
	// childNode->setCheckState(0, Qt::Checked);
}

void QGraphicsRepTreeWidgetItem::childRemoved(IGraphicRepFactory *child) {
	QGraphicsRepTreeWidgetItem *childNode = nullptr;
	long i=0;
	while (childNode==nullptr && i<childCount()) {
		QGraphicsRepTreeWidgetItem* currentChild =
				dynamic_cast<QGraphicsRepTreeWidgetItem*>(this->child(i));
		if (currentChild!=nullptr && currentChild->getRep()->data()->graphicRepFactory()==child) {
			childNode = currentChild;
		}
		i++;
	}
	if (childNode == nullptr)
		return;
	removeChild(childNode);
	if (childNode->checkState(0)!=Qt::Unchecked) {
		m_view->hideRep(childNode->getRep());
	}
	childNode->deleteLater();
}

void QGraphicsRepTreeWidgetItem::connectChildRep(AbstractGraphicRep *childRep){
	// MZR 15072021
	if((dynamic_cast<SliceRep *>(childRep) != nullptr)
			|| (dynamic_cast<DatasetRep *>(childRep) != nullptr)
			|| (dynamic_cast<Dataset3DSliceRep*>(childRep) != nullptr)
			|| (dynamic_cast<StackLayerRGTRep *>(childRep) != nullptr)
			|| (dynamic_cast<RGBLayerRGTRep*>(childRep) != nullptr)
			|| (dynamic_cast<LayerRGTRep*>(childRep) != nullptr)
			|| (dynamic_cast<LayerRGTRepOnSlice*>(childRep) != nullptr)
			// random
			|| (dynamic_cast<RandomRep*>(childRep) != nullptr)
			|| (dynamic_cast<WellBoreRepOnRandom*>(childRep) != nullptr)
			|| (dynamic_cast<WellPickRepOnRandom*>(childRep) != nullptr)

			|| (dynamic_cast<WellBoreRepOnSlice*>(childRep) != nullptr)
			|| (dynamic_cast<WellBoreRepOn3D*>(childRep) != nullptr)
			|| (dynamic_cast<WellBoreRepOnMap*>(childRep) != nullptr)
			|| (dynamic_cast<WellPickRepOnSlice*>(childRep) != nullptr))
	{
		connect(childRep, SIGNAL(deletedRep(AbstractGraphicRep *)), this, SLOT(deletedRep(AbstractGraphicRep *)));
	}
}

QGraphicsRepTreeWidgetItem* QGraphicsRepTreeWidgetItem::generateChild(
		IGraphicRepFactory *child) {
	if (child == nullptr) {
		return nullptr;
	}
	AbstractGraphicRep *childRep = child->rep(m_view->viewType(), m_view);
	if (childRep == nullptr)
		return nullptr;

	connectChildRep(childRep);

	return new QGraphicsRepTreeWidgetItem(childRep, child, m_view, this);
}

// MZR 16072021
void QGraphicsRepTreeWidgetItem::deletedRep(AbstractGraphicRep * rep){
	QTreeWidgetItem * pTreeWidgetItem = QInnerViewTreeWidgetItem::findRepNode(rep,this);
	QGraphicsRepTreeWidgetItem *pGraphicWItem = dynamic_cast<QGraphicsRepTreeWidgetItem *>(pTreeWidgetItem);
	if(pGraphicWItem != nullptr){
		emit repDeleted(rep);
		removeChild(pGraphicWItem);
		pGraphicWItem->setRep(nullptr);
	}
}

void QGraphicsRepTreeWidgetItem::setRep(AbstractGraphicRep * rep){
	m_rep = rep;
}

const AbstractGraphicRep* QGraphicsRepTreeWidgetItem::getRep() const {
	return m_rep;
}

AbstractGraphicRep* QGraphicsRepTreeWidgetItem::getRep() {
	return m_rep;
}

void QGraphicsRepTreeWidgetItem::updateItemWithDecorator() {
	if (!m_decorator.isNull()) {
		m_decorator->decorate(this);
	}
}
