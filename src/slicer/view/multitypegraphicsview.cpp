#include "multitypegraphicsview.h"
#include <QPushButton>
#include <QToolBar>
#include <QActionGroup>
#include <QMenu>
#include <QDebug>
#include <QGraphicsScene>
#include <QContextMenuEvent>
#include <QTreeWidgetItem>
#include <cmath>

#include "graphicsutil.h"
#include "basemapview.h"
#include "stackbasemapview.h"
#include "randomlineview.h"
#include "wellbore.h"
#include "selectrandomcreationmode.h"
#include "qinnerviewtreewidgetitem.h"
#include "abstractinnerview.h"


MultiTypeGraphicsView::MultiTypeGraphicsView(WorkingSetManager *factory,
		QString uniqueName, QWidget *parent) :
		AbstractGraphicsView(factory, uniqueName, parent) {

	QPushButton *addInternalView = GraphicsUtil::generateToobarButton(
			":/slicer/icons/AddView.svg", "Add view", toolBar());
	addInternalView->setIconSize(QSize(32, 32));
	addInternalView->setFixedSize(32,32);
	addInternalView->setDefault(false);
	addInternalView->setAutoDefault(false);
	connect(addInternalView, SIGNAL(clicked()), this, SLOT(openAddViewMenu()));

	m_addViewMenu = new QMenu(this);

	QActionGroup *actions1 = new QActionGroup(m_addViewMenu);
	actions1->addAction(
			m_addViewMenu->addAction(QIcon(":/slicer/icons/map_gray.png"),
					tr("BaseMap")))->setData(ViewType::BasemapView);
	actions1->addAction(
			m_addViewMenu->addAction(QIcon(":/slicer/icons/inline_gray.png"),
					tr("Inline")))->setData(ViewType::InlineView);
	actions1->addAction(
			m_addViewMenu->addAction(QIcon(":/slicer/icons/xline_gray.png"),
					tr("Xline")))->setData(ViewType::XLineView);
	actions1->addAction(
			m_addViewMenu->addAction(QIcon(":/slicer/icons/3d_gray.png"), tr("3D View")))->setData(
			ViewType::View3D);
	actions1->addAction(
			m_addViewMenu->addAction(QIcon(":/slicer/icons/random_gray.png"), tr("Random View")))->setData(
			ViewType::RandomView);
	connect(actions1, SIGNAL(triggered(QAction*)), SLOT(addView(QAction*)));

	m_toolbar->addWidget(addInternalView);



}

void MultiTypeGraphicsView::addView(QAction *action) {

	if (action->data().toInt() == ViewType::BasemapView) {
		registerView(generateView(ViewType::BasemapView, false));
	} else if (action->data().toInt() == ViewType::InlineView) {
		registerView(generateView(ViewType::InlineView, false));
	} else if (action->data().toInt() == ViewType::XLineView) {
		registerView(generateView(ViewType::XLineView, false));
	} else if (action->data().toInt() == ViewType::View3D) {
		registerView(generateView(ViewType::View3D, false));
	} else if (action->data().toInt() == ViewType::RandomView) {
		selectRandomActionWithUI();
	}
}

void MultiTypeGraphicsView::selectRandomActionWithUI() {
	// create dialog
	SelectRandomCreationMode dialog(m_currentManager, this);
	int result = dialog.exec();

	if (result==QDialog::Accepted) {
		addRandomFromWellBore(dialog.selectedWellBores(), dialog.wellMargin());
	}
}

void MultiTypeGraphicsView::addRandomFromWellBore(QList<WellBore*> wells, double margin) {
	QPolygonF poly;
	bool isFirstSegmentSet = false;
	for (WellBore* well : wells) {
		const Deviations& deviations = well->deviations();
		for (std::size_t idx=0; idx<deviations.xs.size(); idx++) {
			poly << QPointF(deviations.xs[idx], deviations.ys[idx]);
			if (!isFirstSegmentSet && poly.size()>=2) {
				QPolygonF tmpPoly(poly);

				QPointF second = poly[1];
				QPointF first = poly[0];
				QPointF vect = first - second;
				double dist = std::sqrt(vect.x()*vect.x() + vect.y()*vect.y());
				if (dist!=0.0) {
					vect = vect / dist * margin;
				} else {
					vect = QPointF(-margin, 0);
				}
				QPointF marginPoint = first + vect;

				poly.clear();
				poly << marginPoint;
				poly << tmpPoly;
				isFirstSegmentSet = true;
			}
		}
	}
	if (poly.size()>1) {
		QPointF last = poly[poly.size()-1];
		QPointF beforeLast = poly[poly.size()-2];
		QPointF vect = last - beforeLast;
		double dist = std::sqrt(vect.x()*vect.x() + vect.y()*vect.y());
		if (dist!=0) {
			vect = vect / dist * margin;
		} else {
			vect = QPointF(margin, 0);
		}
		QPointF marginPoint = last + vect;
		poly << marginPoint;

		RandomLineView* random = createRandomView(poly);
		random->setDisplayDistance(0.1);

		// change name
		QStringList wellNames;
		for (WellBore* well : wells) {
			wellNames << well->name();
		}
		QString realName = "rd : " + wellNames.join(", ");
		changeViewName(random, realName);
	}
}

RandomLineView* MultiTypeGraphicsView::createRandomView(QPolygonF polygon,eRandomType eType) {
	QString newUniqueName = uniqueName() + "_view" + QString::number(getNewUniqueId());
	RandomLineView* randomView = new RandomLineView(polygon, ViewType::RandomView, newUniqueName,eType);
	registerView(randomView);
	return randomView;
}

void MultiTypeGraphicsView::unregisterRandomView(RandomLineView *pView) {
    if(pView != nullptr){
        unregisterView(pView);
    }
}

AbstractInnerView* MultiTypeGraphicsView::createInnerView(ViewType viewType) {
	AbstractInnerView* innerView = nullptr;
	if (viewType == ViewType::BasemapView) {
		innerView = generateView(ViewType::BasemapView, false);
		registerView(innerView);
	} else if (viewType == ViewType::StackBasemapView) {
		innerView = generateView(ViewType::StackBasemapView, false);
		registerView(innerView);
	} else if (viewType == ViewType::InlineView) {
		innerView = generateView(ViewType::InlineView, false);
		registerView(innerView);
	} else if (viewType == ViewType::XLineView) {
		innerView = generateView(ViewType::XLineView, false);
		registerView(innerView);
	} else if (viewType == ViewType::View3D) {
		innerView = generateView(ViewType::View3D, false);
		registerView(innerView);
	}
	return innerView;
}

void MultiTypeGraphicsView::setInnerViewDefaultName(AbstractInnerView* innerView, QString newName) {
	if (innerView == nullptr) {
		return;
	}
	int itemIdx = 0;
	QInnerViewTreeWidgetItem* treeItem = nullptr;
	while (itemIdx < m_rootItem->childCount() && treeItem==nullptr) {
		if (QInnerViewTreeWidgetItem *it =
				dynamic_cast<QInnerViewTreeWidgetItem*>(m_rootItem->child(itemIdx))) {
			if (innerView==it->innerView()) {
				treeItem = it;
			}
		}
		itemIdx ++;
	}

	if (treeItem!=nullptr) {
		innerView->setDefaultTitle(newName);
		treeItem->setData(0, Qt::DisplayRole, innerView->windowTitle());
	}
}

void MultiTypeGraphicsView::openAddViewMenu() {
	m_addViewMenu->exec(QCursor::pos());
}

MultiTypeGraphicsView::~MultiTypeGraphicsView() {

}
