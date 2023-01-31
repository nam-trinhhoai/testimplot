#include "multiview.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include "mathutils.h"
#include "viewportmanager.h"
#include "cameracontroller.h"


/*<file alias="ViewportGeneric.qml">slicer/multiview/Qml/ViewportGeneric.qml</file>
	<file alias="MultiView.qml">slicer/multiview/Qml/MultiView.qml</file>
	<file alias="View3D.qml">slicer/multiview/Qml/View3D.qml</file>
	<file alias="MultiFrameGraph.qml">slicer/multiview/Qml/MultiFrameGraph.qml</file>
	<file alias="SliceSceneFrameGraph.qml">slicer/multiview/Qml/SliceSceneFrameGraph.qml</file>
	<file alias="SubViewport.qml">slicer/multiview/Qml/SubViewport.qml</file>
	<file alias="ViewportSplitter.qml">slicer/multiview/Qml/ViewportSplitter.qml</file>
	<file alias="ViewportHSplitter.qml">slicer/multiview/Qml/ViewportHSplitter.qml</file>
	<file alias="FreeSliceViewportHeaderBar.qml">slicer/multiview/Qml/FreeSliceViewportHeaderBar.qml</file>
	<file alias="FreeSliceSlider.qml">slicer/multiview/Qml/FreeSliceSlider.qml</file>
	<file alias="AxesGnomon.qml">slicer/multiview/Qml/AxesGnomon.qml</file>
	*/

namespace {
QObject *mathUtilsSingletonProvider(QQmlEngine *, QJSEngine *)
{
    return new MathUtils();
}

}


MultiView::MultiView(bool restictToMonoTypeSplit, QString uniqueName,QWidget *parent) :
KDDockWidgets::DockWidget(uniqueName) //, KDDockWidgets::MainWindowOption_None, parent)
{
	setMinimumSize(QSize(1600, 1000));
	setAttribute(Qt::WA_DeleteOnClose);
		//setDockNestingEnabled(true);

	m_sceneMultiManager = new SceneMultiManager(this);

	/*m_quickview = new QQuickView();//this
	m_quickview->engine()->rootContext()->setContextProperty("multiViewQt", this);
	//m_quickview->setPersistentOpenGLContext(true);
	m_quickview->setPersistentGraphics(true);
	m_quickview->setResizeMode(QQuickView::SizeRootObjectToView);
	//m_quickview->setMinimumSize(QSize(1200, 1000));
	m_quickview->setColor("#FF00000");

	 connect(m_quickview, &QQuickView::statusChanged, this, &MultiView::onQMLReady, Qt::QueuedConnection);

	 qmlRegisterType<CameraController>("Murat", 1, 0, "CameraController");
	 qmlRegisterType<ViewportManager>("Murat", 1, 0, "ViewportManager");
	 //qmlRegisterType<ViewportManager>("Murat", 1, 0, "SceneMultiManager");

	 m_quickview->rootContext()->setContextProperty("sceneMultiManager", m_sceneMultiManager);

	 qmlRegisterSingletonType<MathUtils>("Murat", 1, 0, "MathUtils", &mathUtilsSingletonProvider);
	m_quickview->setSource(QUrl("qrc:/MultiView.qml"));

	QWidget *mainWidget = new QWidget(this);

	QVBoxLayout *mainWidgetLayout = new QVBoxLayout(mainWidget);
	mainWidgetLayout->setContentsMargins(0,0,0,0);
	QHBoxLayout* layH = new QHBoxLayout();
	layH->addWidget(QWidget::createWindowContainer(m_quickview, mainWidget), 1);

	mainWidgetLayout->addLayout(layH,1);


	setWidget(mainWidget);*/
}

MultiView::~MultiView()
{

}

void MultiView::onQMLReady(QQuickView::Status status)
{

	if (status!=QQuickView::Ready) {
		qDebug()<<" qml not ready "<<status;
			return;
		}
	//	disconnect(m_quickview, &QQuickView::statusChanged, this, &ViewQt3D::onQMLReady);

	qDebug()<<" on qml ready";
}
