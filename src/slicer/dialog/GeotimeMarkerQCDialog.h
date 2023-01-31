/*
 * GeotimeMarkerQCDialog.h
 *
 *  Created on: Jan 21, 2021
 *      Author: l0483271
 */

#ifndef SRC_SLICER_DIALOG_GEOTIMEMARKERQCDIALOG_H_
#define SRC_SLICER_DIALOG_GEOTIMEMARKERQCDIALOG_H_

#include <QDialog>
#include <QtCharts/QChartGlobal>
#include <QtCharts/QChartView>
#include <QtCharts/QChart>
#include <QtCharts/QBarSeries>
#include <QtCharts/QValueAxis>

#include "viewutils.h"

class QTableWidget;
class QListWidget;
class WorkingSetManager;
class Seismic3DAbstractDataset;
class Marker;
class WellPick;
class WellBore;
class MtLengthUnit;
class QMouseEvent;
class GeotimeMarkerQCDialog;

class ChartView : public QChartView {
public:
	ChartView(QChart *chart, GeotimeMarkerQCDialog* parentDialog);
	virtual ~ChartView();
protected:
	void mousePressEvent(QMouseEvent *event) override;
	void mouseReleaseEvent(QMouseEvent *event) override;

private:
	GeotimeMarkerQCDialog* m_parentDialog;
};

class GeotimeMarkerQCDialog : public QDialog {
	Q_OBJECT
public:
	GeotimeMarkerQCDialog(QWidget *parent, WorkingSetManager *currentManager, Seismic3DAbstractDataset* dataset, int channel,
			const MtLengthUnit* depthLengthUnit);
	virtual ~GeotimeMarkerQCDialog();

	void mousePressAction1(QMouseEvent *event);
	void mouseReleaseAction1(QMouseEvent *event);

public slots:
	void setDepthLengthUnit(const MtLengthUnit* depthLengthUnit);

signals:
	void choosedPicks(QList<WellBore*> choosenWells, QList<int> geotime, QList<int> mds);

private:
	void markerChanged(int row);
	//Seismic3DAbstractDataset* selectDataset(bool onlyCpu);
	void computeHistogram(Marker* selectedMarker,
			Seismic3DAbstractDataset* dataset, int channel, SampleUnit sampleUnit);
	void computeSelection();
	void choosePicksSlot();

	double convertDepthForDisplay(double oriVal);

	QTableWidget *m_wellsTable;
	WorkingSetManager *m_currentManager;
	QListWidget* m_markersListWidget;
	QList<WellPick*> m_wellPicks;
	QList<Marker*> m_markers;
	Seismic3DAbstractDataset* m_dataset;
	int m_channel;
	QChart* m_chart;
	ChartView* m_chartView;
	QBarSeries* m_histoSeries = nullptr;
	QValueAxis *m_axisValueX = nullptr;
	QValueAxis *m_axisY = nullptr;

	QList<WellBore*> m_bores;
	QList<int> m_geotimesList;
	QList<int> m_mdList;
	QList<int> m_twtList;
	QList<WellBore*> m_selectedBores;
	QList<int> m_selectedGeotimesList;
	QList<int> m_selectedMdList;
	QList<int> m_selectedTwtList;
	int m_minGt = 999999;
	int m_maxGt = -999999;
	static const int m_histoSize = 30;
	int m_histo[m_histoSize];
	float m_minSelectIndex = 0;
	float m_maxSelectIndex = 0;
	float m_minSelectWidget = 0;
	float m_maxSelectWidget = 0;

	const MtLengthUnit* m_depthLengthUnit = nullptr;
};

#endif /* SRC_SLICER_DIALOG_GEOTIMEMARKERQCDIALOG_H_ */
