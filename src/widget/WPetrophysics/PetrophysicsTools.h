/*
 *
 *
 *  Created on: 18 Jul 2022
 *      Author: l0359127
 */

#ifndef NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PETROPHYSICSTOOLS_H_
#define NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PETROPHYSICSTOOLS_H_

#include "PlotWellLog.h"
#include "CrossPlotWellLogs.h"
#include "CrossPlotRegression.h"	
#include "CrossPlotSplineRegression.h"	
#include "CrossPlotPowerRegression.h"
#include "PointsSelectionDragRect.h"
#include "PointsSelectionMultipleDragRect.h"
#include "MultipleIntervalsCrossPlot.h"
#include "MultipleIntervalsCrossPlotRegression.h"
#include "MultipleIntervalsCrossPlotDragRect.h"
#include "PlotSingleWellLog.h"
#include "SelectLogInterval.h"
#include "SelectMultipleIntervals.h"
#include "HistogramPlotWellLog.h"
#include "PlotHistogramCurve.h"
#include "CrossPlotWellLogsWithHistogram.h"
#include "MultipleIntervalsCrossPlotWithHistogram.h"
#include "MultipleIntervalsCrossPlotWithHistogramRegression.h"
#include "PlotMultipleWellLogs.h"
#include "PlotMultipleWellBores.h"
#include "PlotWellLogShaded.h"
#include "PlotWellLogInteractive.h"
#include "InteractiveHistogram.h"
#include "InteractiveCrossPlot.h"
#include "InteractiveCrossPlotWithHistogram.h"
#include "InteractiveMultipleWindows.h"	

#include "PlotSeismic.h"
#include "PlotMultipleSeismicExtractionsAndWellLogs.h"
#include "PlotWithMultipleKeys.h"

#include "DragCircle.h"	
#include "DragEllipse.h"	
#include "BezierCurve.h"	

#include "OptimizeLongCrosshairRenderSpeed.h"

#include "workingsetmanager.h"
#include "DataSelectorDialog.h"
#include "geotimegraphicsview.h"
#include "folderdata.h"
#include "wellhead.h"
#include "wellbore.h"

#include <QMainWindow>
#include <QProcess>
#include <QTextCursor>
#include <QFile>
#include <QtWidgets>

QT_BEGIN_NAMESPACE
class QAction;
class QActionGroup;
class QLabel;
class QMenu;
QT_END_NAMESPACE


class PetrophysicsTools : public QMainWindow
{
    Q_OBJECT

public:
    PetrophysicsTools();

protected:
#ifndef QT_NO_CONTEXTMENU
    void contextMenuEvent(QContextMenuEvent *event) override;
#endif // QT_NO_CONTEXTMENU

private slots:
	void importDatabase();
	void viewCrossPlot();
	void viewCrossPlotRegression();
	void viewCrossPlotSplineRegression();
	void viewCrossPlotPowerRegression();	
	void viewPointsSelectionDragRect();
	void viewPointsSelectionMultipleDragRect();
	void viewMultipleIntervalsCrossPlot();
	void viewMultipleIntervalsCrossPlotRegression();
	void viewMultipleIntervalsCrossPlotDragRect();
	void viewPlotSingleWellLog();
	void viewSelectLogInterval();
	void viewSelectMultipleIntervals();
	void viewHistogramPlotWellLog();
	void viewPlotHistogramCurve();
	void viewWellLog();
	void viewCrossPlotWellLogsWithHistogram();
	void viewMultipleIntervalsCrossPlotWithHistogram();
	void viewMultipleIntervalsCrossPlotWithHistogramRegression();
	void viewPlotMultipleWellLogs();
	void viewPlotMultipleWellBores();
	void viewPlotWellLogShaded();
	void viewPlotWellLogInteractive();	
	void viewInteractiveHistogram();
	void viewInteractiveCrossPlot();	
	void viewInteractiveCrossPlotWithHistogram();	
	void viewInteractiveMultipleWindows();	

	void viewPlotSeismic();
	void viewPlotMultipleSeismicExtractionsAndWellLogs();
	void viewPlotWithMultipleKeys();

	void viewDragCircle();
	void viewDragEllipse();
	void viewBezierCurve();

	void viewOptimizeLongCrosshairRenderSpeed();
	
private:
	bool dataImported = false;
	WorkingSetManager* m_manager;

	void createActions();
	void createMenus();

	QMenu *dataMenu;
	QMenu *visualizationMenu;
	QMenu *seismicVisualizationMenu;
	QMenu *canvas2dMenu;
	QMenu *optimizationMenu;
    
    QAction *importDatabaseAct;
	QAction *viewCrossPlotAct;
	QAction *viewCrossPlotRegressionAct;
	QAction *viewCrossPlotSplineRegressionAct;
	QAction *viewCrossPlotPowerRegressionAct;
	QAction *viewPointsSelectionDragRectAct;
	QAction *viewPointsSelectionMultipleDragRectAct;
	QAction *viewMultipleIntervalsCrossPlotAct;
	QAction *viewMultipleIntervalsCrossPlotRegressionAct;
	QAction *viewMultipleIntervalsCrossPlotDragRectAct;
	QAction *viewPlotSingleWellLogAct;
	QAction *viewSelectLogIntervalAct;
	QAction *viewSelectMultipleIntervalsAct;
	QAction *viewWellLogAct;
	QAction *viewHistogramPlotWellLogAct;
	QAction *viewPlotHistogramCurveAct;
	QAction *viewCrossPlotWellLogsWithHistogramAct;
	QAction *viewMultipleIntervalsCrossPlotWithHistogramAct;
	QAction *viewMultipleIntervalsCrossPlotWithHistogramRegressionAct;
	QAction *viewPlotMultipleWellLogsAct;
	QAction *viewPlotMultipleWellBoresAct;
	QAction *viewPlotWellLogShadedAct;
	QAction *viewPlotWellLogInteractiveAct;
	QAction *viewInteractiveHistogramAct;	
	QAction *viewInteractiveCrossPlotAct;	
	QAction *viewInteractiveCrossPlotWithHistogramAct;
	QAction *viewInteractiveMultipleWindowsAct;	

	QAction *viewPlotSeismicAct;
	QAction *viewPlotMultipleSeismicExtractionsAndWellLogsAct;
	QAction *viewPlotWithMultipleKeysAct;

	QAction *viewDragCircleAct;
	QAction *viewDragEllipseAct;
	QAction *viewBezierCurveAct;

	QAction *viewOptimizeLongCrosshairRenderSpeedAct;
};

#endif // NEXTVISION_SRC_WIDGET_WPETROPHYSICS_PETROPHYSICSTOOLS_H_
