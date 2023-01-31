/*
 *
 *
 *  Created on: 18 Jul 2022
 *      Author: l0359127
 */

#include "PetrophysicsTools.h"

PetrophysicsTools::PetrophysicsTools()
{
    //QWidget *widget = new QWidget;
    //setCentralWidget(widget);

    createActions();
    createMenus();

    //QString message = tr("Tools for geomechanical simulation at well scale");
    //statusBar()->showMessage(message);

    setWindowTitle(tr("Petrophysics"));
    setMinimumSize(160, 160);
    resize(1012, 683);
}

#ifndef QT_NO_CONTEXTMENU
void PetrophysicsTools::contextMenuEvent(QContextMenuEvent *event)
{
    QMenu menu(this);
	menu.addAction(importDatabaseAct);
    menu.exec(event->globalPos());
}
#endif // QT_NO_CONTEXTMENU

void PetrophysicsTools::importDatabase()
{
	WorkingSetManager* manager = new WorkingSetManager(this);

	DataSelectorDialog* dialog = new DataSelectorDialog(this, manager);
	dialog->resize(550*2, 950);
	int code = dialog->exec();

	if (code==QDialog::Accepted) {
		m_manager = manager;
		dataImported = true;
	}
}

// Interaction between windows
void PetrophysicsTools::viewInteractiveMultipleWindows()
{
	if (dataImported)
	{
		InteractiveMultipleWindows *w = new InteractiveMultipleWindows(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot cross-plot with histogram interactively
void PetrophysicsTools::viewInteractiveCrossPlotWithHistogram()
{
	if (dataImported)
	{
		InteractiveCrossPlotWithHistogram *w = new InteractiveCrossPlotWithHistogram(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot cross-plot interactively
void PetrophysicsTools::viewInteractiveCrossPlot()
{
	if (dataImported)
	{
		InteractiveCrossPlot *w = new InteractiveCrossPlot(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot histogram interactively
void PetrophysicsTools::viewInteractiveHistogram()
{
	if (dataImported)
	{
		InteractiveHistogram *w = new InteractiveHistogram(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot well-log interactively
void PetrophysicsTools::viewPlotWellLogInteractive()
{
	if (dataImported)
	{
		PlotWellLogInteractive *w = new PlotWellLogInteractive(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot shaded well-log
void PetrophysicsTools::viewPlotWellLogShaded()
{
	if (dataImported)
	{
		PlotWellLogShaded *w = new PlotWellLogShaded(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot multiple wellbores
void PetrophysicsTools::viewPlotMultipleWellBores()
{
	if (dataImported)
	{
		PlotMultipleWellBores *w = new PlotMultipleWellBores(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot multiple well logs
void PetrophysicsTools::viewPlotMultipleWellLogs()
{
	if (dataImported)
	{
		PlotMultipleWellLogs *w = new PlotMultipleWellLogs(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View multiple intervals cross-plot with histogram and regression
void PetrophysicsTools::viewMultipleIntervalsCrossPlotWithHistogramRegression()
{	
	if (dataImported)
	{
		MultipleIntervalsCrossPlotWithHistogramRegression *w = new MultipleIntervalsCrossPlotWithHistogramRegression(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View multiple intervals cross-plot with histogram
void PetrophysicsTools::viewMultipleIntervalsCrossPlotWithHistogram()
{	
	if (dataImported)
	{
		MultipleIntervalsCrossPlotWithHistogram *w = new MultipleIntervalsCrossPlotWithHistogram(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot with histogram
void PetrophysicsTools::viewCrossPlotWellLogsWithHistogram()
{	
	if (dataImported)
	{
		CrossPlotWellLogsWithHistogram *w = new CrossPlotWellLogsWithHistogram(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log histogram curve
void PetrophysicsTools::viewPlotHistogramCurve()
{	
	if (dataImported)
	{
		PlotHistogramCurve *w = new PlotHistogramCurve(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log histogram
void PetrophysicsTools::viewHistogramPlotWellLog()
{	
	if (dataImported)
	{
		HistogramPlotWellLog *w = new HistogramPlotWellLog(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View multiple intervals cross-plot with points selector
void PetrophysicsTools::viewMultipleIntervalsCrossPlotDragRect()
{
	if (dataImported)
	{
		MultipleIntervalsCrossPlotDragRect *w = new MultipleIntervalsCrossPlotDragRect(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View multiple intervals cross-plot with regression
void PetrophysicsTools::viewMultipleIntervalsCrossPlotRegression()
{
	if (dataImported)
	{
		MultipleIntervalsCrossPlotRegression *w = new MultipleIntervalsCrossPlotRegression(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View multiple intervals cross-plot
void PetrophysicsTools::viewMultipleIntervalsCrossPlot()
{
	if (dataImported)
	{
		MultipleIntervalsCrossPlot *w = new MultipleIntervalsCrossPlot(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot with multiple points selection tools
void PetrophysicsTools::viewPointsSelectionMultipleDragRect()
{
	if (dataImported)
	{
		PointsSelectionMultipleDragRect *w = new PointsSelectionMultipleDragRect(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot with points selection tools
void PetrophysicsTools::viewPointsSelectionDragRect()
{
	if (dataImported)
	{
		PointsSelectionDragRect *w = new PointsSelectionDragRect(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot with power regression
void PetrophysicsTools::viewCrossPlotPowerRegression()
{
	if (dataImported)
	{
		CrossPlotPowerRegression *w = new CrossPlotPowerRegression(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot with spline regression
void PetrophysicsTools::viewCrossPlotSplineRegression()
{
	if (dataImported)
	{
		CrossPlotSplineRegression *w = new CrossPlotSplineRegression(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot with linear regression
void PetrophysicsTools::viewCrossPlotRegression()
{
	if (dataImported)
	{
		CrossPlotRegression *w = new CrossPlotRegression(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// View well log cross-plot
void PetrophysicsTools::viewCrossPlot()
{
	if (dataImported)
	{
		CrossPlotWellLogs *w = new CrossPlotWellLogs(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Select multiple log intervals
void PetrophysicsTools::viewSelectMultipleIntervals()
{
	if (dataImported)
	{
		SelectMultipleIntervals *w = new SelectMultipleIntervals(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Select a log interval
void PetrophysicsTools::viewSelectLogInterval()
{
	if (dataImported)
	{
		SelectLogInterval *w = new SelectLogInterval(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot a single log
void PetrophysicsTools::viewPlotSingleWellLog()
{
	if (dataImported)
	{
		PlotSingleWellLog *w = new PlotSingleWellLog(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Well log viewer
void PetrophysicsTools::viewWellLog()
{
	PlotWellLog *w = new PlotWellLog(m_manager);
	w->show();
}


// Seismic
// Plot seismic extraction along a wellbore
void PetrophysicsTools::viewPlotSeismic()
{
	if (dataImported)
	{
		PlotSeismic *w = new PlotSeismic(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}


// Plot multiple seismic extractions along a wellbore and its logs
void PetrophysicsTools::viewPlotMultipleSeismicExtractionsAndWellLogs()
{
	if (dataImported)
	{
		PlotMultipleSeismicExtractionsAndWellLogs *w = new PlotMultipleSeismicExtractionsAndWellLogs(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Plot multiple seismic extractions along a wellbore and its logs with depth or time
void PetrophysicsTools::viewPlotWithMultipleKeys()
{
	if (dataImported)
	{
		PlotWithMultipleKeys *w = new PlotWithMultipleKeys(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}


// Canvas 2D
// Drag Circle
void PetrophysicsTools::viewDragCircle()
{
	if (dataImported)
	{
		DragCircle *w = new DragCircle(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Drag Ellipse
void PetrophysicsTools::viewDragEllipse()
{
	if (dataImported)
	{
		DragEllipse *w = new DragEllipse(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Drag Bezier Curve
void PetrophysicsTools::viewBezierCurve()
{
	if (dataImported)
	{
		BezierCurve *w = new BezierCurve(m_manager);
		w->show();
	}
	else
		qDebug() << "Data is not imported yet.";
}

// Optimize long crosshair render speed
void PetrophysicsTools::viewOptimizeLongCrosshairRenderSpeed()
{
	OptimizeLongCrosshairRenderSpeed *w = new OptimizeLongCrosshairRenderSpeed(m_manager);
	w->show();
}


// Create actions
void PetrophysicsTools::createActions()
{
	// Import data from database
	importDatabaseAct = new QAction(tr("Import from database"), this);
    importDatabaseAct->setStatusTip(tr("Import data from database"));
    connect(importDatabaseAct, &QAction::triggered, this, &PetrophysicsTools::importDatabase);

	// Plot a single log
	viewPlotSingleWellLogAct = new QAction(tr("Plot a single log"), this);
    viewPlotSingleWellLogAct->setStatusTip(tr("Plot a single log"));
    connect(viewPlotSingleWellLogAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotSingleWellLog);

	// Select log interval
	viewSelectLogIntervalAct = new QAction(tr("Select log interval"), this);
    viewSelectLogIntervalAct->setStatusTip(tr("Select a log interval"));
    connect(viewSelectLogIntervalAct, &QAction::triggered, this, &PetrophysicsTools::viewSelectLogInterval);

	// Select multiple log intervals
	viewSelectMultipleIntervalsAct = new QAction(tr("Select multiple log intervals"), this);
    viewSelectMultipleIntervalsAct->setStatusTip(tr("Select multiple log intervals"));
    connect(viewSelectMultipleIntervalsAct, &QAction::triggered, this, &PetrophysicsTools::viewSelectMultipleIntervals);

	// Plot multiple well logs
	viewPlotMultipleWellLogsAct = new QAction(tr("Plot multiple logs"), this);
    viewPlotMultipleWellLogsAct->setStatusTip(tr("Plot multiple well logs"));
    connect(viewPlotMultipleWellLogsAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotMultipleWellLogs);

	// Plot multiple well bores
	viewPlotMultipleWellBoresAct = new QAction(tr("Plot multiple wellbores"), this);
    viewPlotMultipleWellBoresAct->setStatusTip(tr("Plot multiple wellbores"));
    connect(viewPlotMultipleWellBoresAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotMultipleWellBores);

	// View well logs cross-plot
	viewCrossPlotAct = new QAction(tr("Cross Plot"), this);
    viewCrossPlotAct->setStatusTip(tr("View well log cross-plot"));
    connect(viewCrossPlotAct, &QAction::triggered, this, &PetrophysicsTools::viewCrossPlot);

	// View well logs cross-plot with linear regression
	viewCrossPlotRegressionAct = new QAction(tr("Cross Plot with Linear Regression"), this);
    viewCrossPlotRegressionAct->setStatusTip(tr("View well log cross-plot with linear regression"));
    connect(viewCrossPlotRegressionAct, &QAction::triggered, this, &PetrophysicsTools::viewCrossPlotRegression);

	// View well logs cross-plot with spline regression
	viewCrossPlotSplineRegressionAct = new QAction(tr("Cross Plot with Spline Regression"), this);
    viewCrossPlotSplineRegressionAct->setStatusTip(tr("View well log cross-plot with spline regression"));
    connect(viewCrossPlotSplineRegressionAct, &QAction::triggered, this, &PetrophysicsTools::viewCrossPlotSplineRegression);

	// View well logs cross-plot with power regression
	viewCrossPlotPowerRegressionAct = new QAction(tr("Cross Plot with Power Regression"), this);
    viewCrossPlotPowerRegressionAct->setStatusTip(tr("View well log cross-plot with power regression"));
    connect(viewCrossPlotPowerRegressionAct, &QAction::triggered, this, &PetrophysicsTools::viewCrossPlotPowerRegression);

	// View points selection using multiple drag rect
	viewPointsSelectionMultipleDragRectAct = new QAction(tr("Cross-Plot with Multiple Points selector"), this);
    viewPointsSelectionMultipleDragRectAct->setStatusTip(tr("View cross-plot with multiple points selection tools"));
    connect(viewPointsSelectionMultipleDragRectAct, &QAction::triggered, this, &PetrophysicsTools::viewPointsSelectionMultipleDragRect);

	// View points selection using drag rect
	viewPointsSelectionDragRectAct = new QAction(tr("Cross-Plot with Points selection"), this);
    viewPointsSelectionDragRectAct->setStatusTip(tr("View cross-plot with points selection tools"));
    connect(viewPointsSelectionDragRectAct, &QAction::triggered, this, &PetrophysicsTools::viewPointsSelectionDragRect);

	// View multiple intervals cross-plot with regression
	viewMultipleIntervalsCrossPlotRegressionAct = new QAction(tr("Multiple Intervals Cross Plot with Regression"), this);
    viewMultipleIntervalsCrossPlotRegressionAct->setStatusTip(tr("View multiple intervals cross-plot with Regression"));
    connect(viewMultipleIntervalsCrossPlotRegressionAct, &QAction::triggered, this, &PetrophysicsTools::viewMultipleIntervalsCrossPlotRegression);

	// View multiple intervals cross-plot
	viewMultipleIntervalsCrossPlotAct = new QAction(tr("Multiple Intervals Cross Plot"), this);
    viewMultipleIntervalsCrossPlotAct->setStatusTip(tr("View multiple intervals cross-plot"));
    connect(viewMultipleIntervalsCrossPlotAct, &QAction::triggered, this, &PetrophysicsTools::viewMultipleIntervalsCrossPlot);

	// View multiple intervals cross-plot with drag rect points selectors
	viewMultipleIntervalsCrossPlotDragRectAct = new QAction(tr("Multiple Intervals Cross Plot with Points Selector"), this);
    viewMultipleIntervalsCrossPlotDragRectAct->setStatusTip(tr("View multiple intervals cross-plot with drag rect points selector"));
    connect(viewMultipleIntervalsCrossPlotDragRectAct, &QAction::triggered, this, &PetrophysicsTools::viewMultipleIntervalsCrossPlotDragRect);

	// View well log histogram
	viewHistogramPlotWellLogAct = new QAction(tr("Log Histogram"), this);
    viewHistogramPlotWellLogAct->setStatusTip(tr("View well log histogram"));
    connect(viewHistogramPlotWellLogAct, &QAction::triggered, this, &PetrophysicsTools::viewHistogramPlotWellLog);

	// View well log histogram curve
	viewPlotHistogramCurveAct = new QAction(tr("Plot histogram curve"), this);
    viewPlotHistogramCurveAct->setStatusTip(tr("View well log histogram curve"));
    connect(viewPlotHistogramCurveAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotHistogramCurve);		

	// View cross-plot well logs with histogram
	viewCrossPlotWellLogsWithHistogramAct = new QAction(tr("Cross Plot with Histogram"), this);
    viewCrossPlotWellLogsWithHistogramAct->setStatusTip(tr("View well log cross-plot with histogram"));
    connect(viewCrossPlotWellLogsWithHistogramAct, &QAction::triggered, this, &PetrophysicsTools::viewCrossPlotWellLogsWithHistogram);

	// View multiple intervals cross-plot with histogram
	viewMultipleIntervalsCrossPlotWithHistogramAct = new QAction(tr("Multiple Intervals Cross Plot with Histogram"), this);
    viewMultipleIntervalsCrossPlotWithHistogramAct->setStatusTip(tr("View multiple intervals cross-plot with histogram"));
    connect(viewMultipleIntervalsCrossPlotWithHistogramAct, &QAction::triggered, this, &PetrophysicsTools::viewMultipleIntervalsCrossPlotWithHistogram);

	// View multiple intervals cross-plot with histogram and regression
	viewMultipleIntervalsCrossPlotWithHistogramRegressionAct = new QAction(tr("Multiple Intervals Cross Plot with Histogram and Regression"), this);
    viewMultipleIntervalsCrossPlotWithHistogramRegressionAct->setStatusTip(tr("View multiple intervals cross-plot with histogram and regression"));
    connect(viewMultipleIntervalsCrossPlotWithHistogramRegressionAct, &QAction::triggered, this, &PetrophysicsTools::viewMultipleIntervalsCrossPlotWithHistogramRegression);

	// View plot shaded well log
	viewPlotWellLogShadedAct = new QAction(tr("Shaded plot"), this);
    viewPlotWellLogShadedAct->setStatusTip(tr("View well log with shaded effect"));
    connect(viewPlotWellLogShadedAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotWellLogShaded);

	// View plot well log interactive
	viewPlotWellLogInteractiveAct = new QAction(tr("Interactive shaded plot"), this);
    viewPlotWellLogInteractiveAct->setStatusTip(tr("View well log with interactive effect"));
    connect(viewPlotWellLogInteractiveAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotWellLogInteractive);

	// View interactive histogram
	viewInteractiveHistogramAct = new QAction(tr("Interactive histogram"), this);
    viewInteractiveHistogramAct->setStatusTip(tr("View interactive histogram"));
    connect(viewInteractiveHistogramAct, &QAction::triggered, this, &PetrophysicsTools::viewInteractiveHistogram);

	// View interactive cross-plot
	viewInteractiveCrossPlotAct = new QAction(tr("Interactive cross-plot"), this);
    viewInteractiveCrossPlotAct->setStatusTip(tr("View interactive cross-plot"));
    connect(viewInteractiveCrossPlotAct, &QAction::triggered, this, &PetrophysicsTools::viewInteractiveCrossPlot);

	// View interactive cross-plot with histogram
	viewInteractiveCrossPlotWithHistogramAct = new QAction(tr("Interactive cross-plot with histogram"), this);
    viewInteractiveCrossPlotWithHistogramAct->setStatusTip(tr("View interactive cross-plot with histogram"));
    connect(viewInteractiveCrossPlotWithHistogramAct, &QAction::triggered, this, &PetrophysicsTools::viewInteractiveCrossPlotWithHistogram);

	// View interactive plot with multiple windows
	viewInteractiveMultipleWindowsAct = new QAction(tr("Interactive plot with multiple windows"), this);
    viewInteractiveMultipleWindowsAct->setStatusTip(tr("View interactive plot with multiple windows"));
    connect(viewInteractiveMultipleWindowsAct, &QAction::triggered, this, &PetrophysicsTools::viewInteractiveMultipleWindows);

	// Well-log viewer (old version)
	viewWellLogAct = new QAction(tr("Well-log viewer"), this);
    viewWellLogAct->setStatusTip(tr("Old version of well-log viewer"));
    connect(viewWellLogAct, &QAction::triggered, this, &PetrophysicsTools::viewWellLog);

	// Seismic
	// Plot a single seismic extraction
	viewPlotSeismicAct = new QAction(tr("Plot a single seismic extraction"), this);
    viewPlotSeismicAct->setStatusTip(tr("Plot a single seismic extraction along a wellbore"));
    connect(viewPlotSeismicAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotSeismic);

	// Plot mutlplie seismic extractions and well logs
	viewPlotMultipleSeismicExtractionsAndWellLogsAct = new QAction(tr("Plot multiple seismic extractions and well logs"), this);
    viewPlotMultipleSeismicExtractionsAndWellLogsAct->setStatusTip(tr("Plot multiple seismic extractions along a wellbore and its well logs"));
    connect(viewPlotMultipleSeismicExtractionsAndWellLogsAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotMultipleSeismicExtractionsAndWellLogs);

	// Plot mutlplie seismic extractions and well logs with multiple keys
	viewPlotWithMultipleKeysAct = new QAction(tr("Plot seismic extractions and logs with depth or time"), this);
    viewPlotWithMultipleKeysAct->setStatusTip(tr("Plot seismic extractions and logs with depth or time"));
    connect(viewPlotWithMultipleKeysAct, &QAction::triggered, this, &PetrophysicsTools::viewPlotWithMultipleKeys);

	// Canvas 2D
	// Drag Circle
	viewDragCircleAct = new QAction(tr("Drag Circle"), this);
    viewDragCircleAct->setStatusTip(tr("Drag Circle to select cross-plot points"));
    connect(viewDragCircleAct, &QAction::triggered, this, &PetrophysicsTools::viewDragCircle);

	// Drag Ellipse
	viewDragEllipseAct = new QAction(tr("Drag Ellipse"), this);
    viewDragEllipseAct->setStatusTip(tr("Drag Ellipse to select cross-plot points"));
    connect(viewDragEllipseAct, &QAction::triggered, this, &PetrophysicsTools::viewDragEllipse);

	// Drag Bezier Curve
	viewBezierCurveAct = new QAction(tr("Drag Bezier Curve"), this);
    viewBezierCurveAct->setStatusTip(tr("Drag Bezier curve to select cross-plot points"));
    connect(viewBezierCurveAct, &QAction::triggered, this, &PetrophysicsTools::viewBezierCurve);

	// Optimization
	viewOptimizeLongCrosshairRenderSpeedAct = new QAction(tr("Optimize Long Crosshair Render Speed"), this);
    viewOptimizeLongCrosshairRenderSpeedAct->setStatusTip(tr("Optimize Long Crosshair Render Speed"));
    connect(viewOptimizeLongCrosshairRenderSpeedAct, &QAction::triggered, this, &PetrophysicsTools::viewOptimizeLongCrosshairRenderSpeed);
}

void PetrophysicsTools::createMenus()
{
	dataMenu = menuBar()->addMenu(tr("Data"));
	dataMenu->addAction(importDatabaseAct);

	visualizationMenu = menuBar()->addMenu(tr("Well Logs"));
	visualizationMenu->addAction(viewPlotSingleWellLogAct);
	visualizationMenu->addAction(viewSelectLogIntervalAct);
	visualizationMenu->addAction(viewSelectMultipleIntervalsAct);
	visualizationMenu->addAction(viewPlotMultipleWellLogsAct);
	visualizationMenu->addAction(viewPlotMultipleWellBoresAct);
	visualizationMenu->addAction(viewCrossPlotAct);
	visualizationMenu->addAction(viewCrossPlotRegressionAct);
	visualizationMenu->addAction(viewCrossPlotSplineRegressionAct);
	visualizationMenu->addAction(viewCrossPlotPowerRegressionAct);
	visualizationMenu->addAction(viewPointsSelectionDragRectAct);
	visualizationMenu->addAction(viewPointsSelectionMultipleDragRectAct);
	visualizationMenu->addAction(viewMultipleIntervalsCrossPlotAct);
	visualizationMenu->addAction(viewMultipleIntervalsCrossPlotRegressionAct);
	visualizationMenu->addAction(viewMultipleIntervalsCrossPlotDragRectAct);	
	visualizationMenu->addAction(viewHistogramPlotWellLogAct);
	visualizationMenu->addAction(viewPlotHistogramCurveAct);
	visualizationMenu->addAction(viewCrossPlotWellLogsWithHistogramAct);
	visualizationMenu->addAction(viewMultipleIntervalsCrossPlotWithHistogramAct);
	visualizationMenu->addAction(viewMultipleIntervalsCrossPlotWithHistogramRegressionAct);
	visualizationMenu->addAction(viewPlotWellLogShadedAct);
	visualizationMenu->addAction(viewPlotWellLogInteractiveAct);
	visualizationMenu->addAction(viewInteractiveHistogramAct);
	visualizationMenu->addAction(viewInteractiveCrossPlotAct);
	visualizationMenu->addAction(viewInteractiveCrossPlotWithHistogramAct);
	visualizationMenu->addAction(viewInteractiveMultipleWindowsAct);
	visualizationMenu->addAction(viewWellLogAct);

	seismicVisualizationMenu = menuBar()->addMenu(tr("Seismic"));
	seismicVisualizationMenu->addAction(viewPlotSeismicAct);
	seismicVisualizationMenu->addAction(viewPlotMultipleSeismicExtractionsAndWellLogsAct);
	seismicVisualizationMenu->addAction(viewPlotWithMultipleKeysAct);

	canvas2dMenu = menuBar()->addMenu(tr("Canvas 2D"));
	canvas2dMenu->addAction(viewDragCircleAct);
	canvas2dMenu->addAction(viewDragEllipseAct);
	canvas2dMenu->addAction(viewBezierCurveAct);

	optimizationMenu = menuBar()->addMenu(tr("Optimization"));
	optimizationMenu->addAction(viewOptimizeLongCrosshairRenderSpeedAct);
}



