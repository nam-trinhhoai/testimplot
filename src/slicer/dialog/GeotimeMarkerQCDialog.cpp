/*
 * GeotimeMarkerQCDialog.cpp
 *
 *  Created on: Jan 21, 2021
 *      Author: Georges Sibille
 *  For one marker on all selected wellbore and one dataset, extract dataset values for picks.
 *  Display an histogram.
 *
 *  qDebug lines have been put in comments and not removed for debug purpose.
 */

#include "GeotimeMarkerQCDialog.h"

#include <QTableWidget>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QListWidget>
#include <QListWidgetItem>
#include <QLabel>
#include <QColor>
#include <QLineEdit>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QGroupBox>
#include <QHeaderView>
#include <QGraphicsItem>
#include <QRect>
#include <QPointF>
#include <QtCharts/QChartGlobal>
#include <QtCharts/QChartView>
#include <QtCharts/QBarSeries>
#include <QtCharts/QBarSet>
#include <QtCharts/QValueAxis>
#include <QtCharts/QBarCategoryAxis>

#include "wellpick.h"
#include "wellbore.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "seismicsurvey.h"
#include "workingsetmanager.h"
#include "marker.h"
#include "folderdata.h"
#include "stringselectordialog.h"
#include "mtlengthunit.h"

ChartView::ChartView(QChart *chart, GeotimeMarkerQCDialog* parentDialog)
    : QChartView(chart), m_parentDialog(parentDialog) {
}

ChartView::~ChartView()
{
}

void ChartView::mousePressEvent(QMouseEvent *event) {
	m_parentDialog->mousePressAction1(event);
}

void ChartView::mouseReleaseEvent(QMouseEvent *event) {
	m_parentDialog->mouseReleaseAction1(event);
}

GeotimeMarkerQCDialog::GeotimeMarkerQCDialog(QWidget *parent, WorkingSetManager *currentManager,
		Seismic3DAbstractDataset* dataset, int channel, const MtLengthUnit* depthLengthUnit):
		QDialog(parent), m_currentManager(currentManager){
	m_depthLengthUnit = depthLengthUnit;
	QVBoxLayout *mainLayout = new QVBoxLayout(this);

	// Data Selection
	QGroupBox* dataGroup = new QGroupBox("Data Selection");
	mainLayout->addWidget(dataGroup, 1);
	QHBoxLayout* dataLayout = new QHBoxLayout(dataGroup);

	m_dataset = dataset;//selectDataset(false);
	if (channel<0 || channel>=m_dataset->dimV()) {
		m_channel = 0;
	} else {
		m_channel = channel;
	}
	QLineEdit *dsNameLE = new QLineEdit();
	dataLayout->addWidget(dsNameLE);
	dsNameLE->setText(m_dataset->name());

	// get markers
	const QList<IData*>& datas = m_currentManager->folders().markers->data();
	for (IData* data : datas) {
		Marker* marker = dynamic_cast<Marker*>(data);
		if (marker!=nullptr) {
			m_markers.push_back(marker);
		}
	}
	// build list
	m_markersListWidget = new QListWidget;
	dataLayout->addWidget(m_markersListWidget);

	for (std::size_t i=0; i<m_markers.size(); i++) {
		Marker* marker = m_markers[i];
		QListWidgetItem* item = new QListWidgetItem(marker->name());

		m_markersListWidget->addItem(item);
	}
	m_markersListWidget->setSelectionMode(QAbstractItemView::SingleSelection);

	connect(m_markersListWidget, &QListWidget::currentRowChanged, this, &GeotimeMarkerQCDialog::markerChanged);

	// Chart
	QGroupBox* chartGroup = new QGroupBox("Chart");
	mainLayout->addWidget(chartGroup, 4);
	QHBoxLayout* qcHLayout = new QHBoxLayout(chartGroup);
	m_chart = new QChart();
	m_chartView = new ChartView(m_chart, this);
	m_chart->legend()->setVisible(false);
	qcHLayout->addWidget(m_chartView, 2);

	QWidget* rightHolder = new QWidget;
	QVBoxLayout* rigthVLayout = new QVBoxLayout;
	rightHolder->setLayout(rigthVLayout);
	qcHLayout->addWidget(rightHolder);

	// Well Table
	m_wellsTable = new QTableWidget(4, 4, this);
	m_wellsTable->setHorizontalHeaderItem(0, new QTableWidgetItem("Wellbore"));
	m_wellsTable->setHorizontalHeaderItem(1, new QTableWidgetItem("Geotime"));
	m_wellsTable->setHorizontalHeaderItem(2, new QTableWidgetItem("MD"));
	m_wellsTable->setHorizontalHeaderItem(3, new QTableWidgetItem("TWT"));
	m_wellsTable->setColumnWidth(0, 160);
	m_wellsTable->setColumnWidth(1, 45);
	m_wellsTable->setColumnWidth(2, 45);
	m_wellsTable->setColumnWidth(3, 45);
	rigthVLayout->addWidget(m_wellsTable, 1);

	QPushButton* choosePicksButton = new QPushButton("Choose picks");
	rigthVLayout->addWidget(choosePicksButton);

	connect(choosePicksButton, &QPushButton::clicked, this, &GeotimeMarkerQCDialog::choosePicksSlot);
}

GeotimeMarkerQCDialog::~GeotimeMarkerQCDialog() {
	// TODO Auto-generated destructor stub
}

void GeotimeMarkerQCDialog::mousePressAction1(QMouseEvent *event) {
	auto const widgetPos = event->localPos();
	auto const scenePos = m_chartView->mapToScene(QPoint(static_cast<int>(widgetPos.x()), static_cast<int>(widgetPos.y())));
	auto const chartItemPos = m_chart->mapFromScene(scenePos);
	auto const valueGivenSeries = m_chart->mapToValue(chartItemPos);
//	qDebug() << "widgetPos:" << widgetPos;
//	qDebug() << "scenePos:" << scenePos;
//	qDebug() << "chartItemPos:" << chartItemPos;
//	qDebug() << "valSeries:" << valueGivenSeries;
//	qDebug() << "mousePressAction " ;
	m_minSelectWidget = widgetPos.x();
	m_minSelectIndex = valueGivenSeries.x();
}

void GeotimeMarkerQCDialog::mouseReleaseAction1(QMouseEvent *event) {
	auto const widgetPos = event->localPos();
	auto const scenePos = m_chartView->mapToScene(QPoint(static_cast<int>(widgetPos.x()), static_cast<int>(widgetPos.y())));
	auto const chartItemPos = m_chart->mapFromScene(scenePos);
	auto const valueGivenSeries = m_chart->mapToValue(chartItemPos);

//	qDebug() << "widgetPos:" << widgetPos;
//	qDebug() << "scenePos:" << scenePos;
//	qDebug() << "chartItemPos:" << chartItemPos;
//	qDebug() << "valSeries:" << valueGivenSeries;
//	qDebug() << "mouseReleaseAction " ;
	if ( valueGivenSeries.x() >= m_minSelectIndex) {
		m_maxSelectWidget = widgetPos.x();
		m_maxSelectIndex = valueGivenSeries.x();
	}
	else {
		m_maxSelectWidget = m_minSelectWidget;
		m_minSelectWidget = widgetPos.x();
		m_maxSelectIndex = m_minSelectIndex;
		m_minSelectIndex = valueGivenSeries.x();
	}
	computeSelection();
}
/*
Seismic3DAbstractDataset* GeotimeMarkerQCDialog::selectDataset(bool onlyCpu) {
	QStringList datasetsNames;
	QList<Seismic3DAbstractDataset*> datasets;

	// fill lists
	for (IData* surveyData : m_currentManager->folders().seismics->data()) {
		if (SeismicSurvey* survey = dynamic_cast<SeismicSurvey*>(surveyData)) {
			for (Seismic3DAbstractDataset* dataset : survey->datasets()) {
				if ((onlyCpu && dynamic_cast<Seismic3DDataset*>(dataset)!=nullptr) || !onlyCpu) {
					datasets.push_back(dataset);
					datasetsNames.push_back(dataset->name());
				}
			}
		}
	}

	QString title = tr("Select Dataset");
	StringSelectorDialog dialog(&datasetsNames, title);

	int code = dialog.exec();

	Seismic3DAbstractDataset* outDataset = nullptr;
	if (code == QDialog::Accepted && dialog.getSelectedIndex()>=0) {
		outDataset = datasets[dialog.getSelectedIndex()];
	}
	return outDataset;
}*/

void GeotimeMarkerQCDialog::markerChanged(int row) {
	//m_markersListWidget->
	Marker* selectedMarker = m_markers[row];
	SampleUnit sampleUnit = m_dataset->cubeSeismicAddon().getSampleUnit();
	computeHistogram(selectedMarker, m_dataset, m_channel, sampleUnit);
}

void GeotimeMarkerQCDialog::computeHistogram(Marker* selectedMarker,
		Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit) {
	const QList<WellPick*>& picks = selectedMarker->wellPicks();

	//qDebug() << "====== BEFORE HISTOGRAM ====== ";
	m_bores.clear();
	m_geotimesList.clear();
	m_mdList.clear();
	m_twtList.clear();
	m_selectedBores.clear();
	m_selectedGeotimesList.clear();
	m_selectedMdList.clear();
	m_selectedTwtList.clear();
	m_minGt = 999999;
	m_maxGt = -999999;
	long rejectionCount = 0;
	for (std::size_t i=0; i<picks.size(); i++) {

		std::pair<RgtSeed, bool> projection = picks[i]->getProjectionOnDataset(dataset, channel, sampleUnit);
		if (projection.second) {
			int gt = projection.first.rgtValue;
			if ( gt < m_minGt)
				m_minGt = gt;
			if ( gt > m_maxGt)
				m_maxGt = gt;
			bool ok;
			double md = picks[i]->wellBore()->getMdFromWellUnit(picks[i]->value(), picks[i]->kindUnit(), &ok);
			double twt;
			bool okTwt;
			if (ok) {
				twt = picks[i]->wellBore()->getDepthFromWellUnit(picks[i]->value(), picks[i]->kindUnit(), SampleUnit::TIME, &okTwt);
			} else {
				md = 0;
				okTwt = false;
			}
			if (!okTwt) {
				twt = 0;
			}
			if (ok) {
				m_bores.push_back(picks[i]->wellBore());
				m_geotimesList.push_back( gt);
				m_mdList.push_back( md);
				m_twtList.push_back( twt); // add even if twt invalid to support depth volumes
			} else {
				rejectionCount++;
			}
			//qDebug() << "Name=" << picks[i]->wellBore()->name() << " Gt= " << gt <<
			//		" KIND=" << picks[i]->kind() << " MD=" << md << "TWT=" << twt;
		}
	}

	if (rejectionCount>0) {
		qDebug() << "GeotimeMarkerQCDialog::computeHistogram rejected " << rejectionCount << " picks because of failed conversions";
	}

	if( m_bores.empty())
		return;

	//qDebug() << "====== COMPUTE HISTOGRAM ====== MinGT" << m_minGt <<	" MaxGt= " << m_maxGt;
	for (int i=0; i< m_histoSize; i++)
		m_histo[i]=0;
	for ( int i = 0; i < m_bores.size(); i++ ) {
		int j = (m_geotimesList[i] - m_minGt) /
				(std::max(1,(m_maxGt - m_minGt + 1) / m_histoSize));
		//qDebug() << "Geotime " << m_bores[i]->name() << " Gt= " <<
		//		m_geotimesList[i] << " Index= " << j;
		if (j >= m_histoSize) {
			qDebug() << "BUG " << m_geotimesList[i]; //TODO
			continue;
		}
		m_histo[j] += 1;
	}

	// Serie
	m_chart->removeAllSeries();
	// We suppose that previous serie have been deleted
	m_histoSeries = new QBarSeries();
	QBarSet* barSet = new QBarSet("x");
	int maxHisto = 0;
	for ( int i = 0; i < m_histoSize; i++ ) {
		*barSet << m_histo[i];
		if ( m_histo[i] > maxHisto )
			maxHisto = m_histo[i];
	}
	m_histoSeries->append(barSet);
	//QString name = QString::fromStdString(me.name);
	m_histoSeries->setName("Histo");
	m_chart->addSeries(m_histoSeries);

	// Axis
//    QStringList categories;
//    for ( int i = 0; i < m_histoSize; i++ )
//    	categories << QString::number(i);
//
//    QtCharts::QBarCategoryAxis *axisX = new QtCharts::QBarCategoryAxis();
//    axisX->append(categories);
//    m_chart->addAxis(axisX, Qt::AlignTop);
//    m_histoSeries->attachAxis(axisX);
	if (m_axisValueX != nullptr )
		delete m_axisValueX;
    m_axisValueX = new QValueAxis();
    m_chart->addAxis(m_axisValueX, Qt::AlignBottom);
    m_axisValueX->setRange(m_minGt, m_maxGt );
    m_axisValueX->setTitleText("Geotime Value");
    m_axisValueX->setTitleVisible(true);

	if (m_axisY != nullptr )
		delete m_axisY;
    m_axisY = new QValueAxis();
    m_chart->addAxis(m_axisY, Qt::AlignLeft);
    m_histoSeries->attachAxis(m_axisY);
    m_axisY->setRange(0, maxHisto );
    m_axisY->setTitleText("Number");
    m_axisY->setTitleVisible(true);
}

void GeotimeMarkerQCDialog::computeSelection() {
	m_wellsTable->clear();
	m_selectedBores.clear();
	m_selectedGeotimesList.clear();
	m_selectedMdList.clear();
	m_selectedTwtList.clear();

	QList<QGraphicsItem *>	items = m_chartView->scene()->items();
	foreach (QGraphicsItem *item, items) {
		QGraphicsRectItem *it = qgraphicsitem_cast<QGraphicsRectItem *>
			(item);
	    if (!it)
	        continue;

	    QRectF rec = it->boundingRect();
	    if ( rec.height() < 2 || rec.width() < 2 || rec.width() > 15 )
	    	continue;
//		qDebug() << "SELECT X= " << rec.x() << " Y= " <<
//				rec.y() << " W= " << rec.width() << " H= " <<
//				rec.height() << " Parent= " <<	it->parentWidget();

		if (  rec.width() > 2 && rec.width() < 40 && rec.height() > 20) {
			QPointF center = rec.center();
			auto const valueFronCenterChart = m_chart->mapToValue(center);
			auto const sceneFromCenter = it->mapToScene(center);
			auto const mapChartFromCenter = m_chart->mapFromScene(sceneFromCenter);
			auto const valueGivenSeries = m_chart->mapToValue(sceneFromCenter);
//			qDebug() << "TEST "	 << " Center:" << center
//					<< " valueFronCenterChart:" << valueFronCenterChart
//					<< " sceneFromCenter:" << sceneFromCenter
//					<< " mapChartFromCenter:" << mapChartFromCenter
//					<< " valueGivenSeries:" << valueGivenSeries;
			if ( m_minSelectIndex <= valueGivenSeries.x() &&
				m_maxSelectIndex >= valueGivenSeries.x() ) {

				it->setBrush(Qt::red);
			}
			else {
				it->setBrush(Qt::blue);
			}
		}
	}

	// Table
//	qDebug() << "SELECT Min= " << m_minSelectIndex << " Max= " <<
//			m_maxSelectIndex;
	QList<int> indexesSelected;
	for ( int i = 0; i < m_bores.size(); i++ ) {
		int j = (m_geotimesList[i] - m_minGt) /
				(std::max(1,(m_maxGt - m_minGt + 1) / 30));
		if (j >= m_minSelectIndex && j <= m_maxSelectIndex) {
			//qDebug() << "Select " << j << " Value= " << m_geotimesList[i];
			indexesSelected.push_back( i);
		}
	}

	if (indexesSelected.empty())
		return;

	m_wellsTable->setRowCount(indexesSelected.size());
	m_wellsTable->setColumnCount(4);
	QStringList tableHeader;
	tableHeader<<"Wellbore"<<"Geotime"<<"MD"<<"TWT";
	m_wellsTable->setHorizontalHeaderLabels(tableHeader);
	m_wellsTable->verticalHeader()->setVisible(false);
	m_wellsTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
	m_wellsTable->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_wellsTable->setSelectionMode(QAbstractItemView::SingleSelection);
	m_wellsTable->setShowGrid(true);
	m_wellsTable->setStyleSheet("QTableView {selection-background-color: red;}");
	//insert data
	int row = 0;
	for (int i : indexesSelected) {
		m_wellsTable->setItem(row, 0, new QTableWidgetItem(m_bores[i]->name()));
		m_wellsTable->setItem(row, 1, new QTableWidgetItem(QString::number(m_geotimesList[i])));
		// display in tab with converted md value
		m_wellsTable->setItem(row, 2, new QTableWidgetItem(QString::number(convertDepthForDisplay(m_mdList[i]))));
		m_wellsTable->setItem(row, 3, new QTableWidgetItem(QString::number(m_twtList[i])));

		m_selectedBores.push_back(m_bores[i]);
		m_selectedGeotimesList.push_back(m_geotimesList[i]);
		m_selectedMdList.push_back(m_mdList[i]);
		m_selectedTwtList.push_back(m_twtList[i]);
		row++;
	}
}

void GeotimeMarkerQCDialog::choosePicksSlot() {
	emit choosedPicks(m_selectedBores, m_selectedGeotimesList, m_selectedMdList);
}

double GeotimeMarkerQCDialog::convertDepthForDisplay(double oriVal) {
	// oriVal is in metre
	return MtLengthUnit::convert(MtLengthUnit::METRE, *m_depthLengthUnit, oriVal);
}

void GeotimeMarkerQCDialog::setDepthLengthUnit(const MtLengthUnit* depthLengthUnit) {
	if (*m_depthLengthUnit != *depthLengthUnit) {
		m_depthLengthUnit = depthLengthUnit;

		int N = std::min(m_wellsTable->rowCount(), (int)(m_selectedMdList.size())); // the two values should be the identical, safety measure
		for (int row = 0; row<N; row++) {
			// display in tab with converted md value
			m_wellsTable->setItem(row, 2, new QTableWidgetItem(QString::number(convertDepthForDisplay(m_selectedMdList[row]))));
		}
	}
}
