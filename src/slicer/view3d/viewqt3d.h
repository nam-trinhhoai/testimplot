#ifndef ViewQt3D_H
#define ViewQt3D_H

#include <QMatrix4x4>
#include <QRayCaster>
#include <Qt3DExtras/QConeMesh>
#include <Qt3DExtras/QSphereMesh>
#include <QTextureImage>
#include <Qt3DRender/QTexture>
#include "abstractinnerview.h"
#include "qrect3d.h"
#include "viewutils.h"
#include "wellborelayer3d.h"
#include "mousetrackingevent.h"
#include "path3d.h"
#include "manager.h"
#include "nurbsmanager.h"
#include "idepthview.h"

#include <QFontDialog>
#include <QColorDialog>

#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DCore/QTransform>
#include <QPropertyAnimation>
#include <QQuickView>
#include <QQuickWidget>
#include <QQueue>
#include <QSpinBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QSlider>
#include <QLineEdit>
#include <QTextEdit>
#include <QPalette>
#include <QPushButton>
#include <QGroupBox>
#include <QGridLayout>
#include <QQmlEngine>
#include <QQmlContext>
#include <QDirectionalLight>
#include <QPointLight>
#include <Qt3DRender/QLayer>

#include "randomview3d.h"



class QVBoxLayout;
class QGridLayout;
class QQuickView;
class QQuickWidget;
class Qt3DRessource;
class QToolButton;
class QSlider;
class QLineEdit;

class RandomLineView;
class MtLengthUnit;
class IToolTipProvider;
class RandomTransformation;

//class GraphEditor_ListBezierPath;

namespace Qt3DRender {
class QCamera;
class QTextureImage;
class QTexture2D;
}

namespace Qt3DCore {
class QEntity;
class QTransform;
}


class InfoTooltip
{

public:

	InfoTooltip( QString inf,QVector3D pos)
	{
		m_infos =inf;
		m_position = pos;

		m_sizePolicy= 14;
		m_zScale = 1.0f;

	}

	InfoTooltip( QString inf,QVector3D pos,int size)
	{
		m_infos =inf;
		m_position = pos;
		m_sizePolicy= size;
		m_zScale = 1.0f;
	}



	InfoTooltip( QString inf,QVector3D pos,int size, int zScale)
		{
			m_infos =inf;
			m_position = pos;
			m_sizePolicy= size;
			m_zScale = zScale;
		}
	InfoTooltip( QString inf,QVector3D pos,int size, int zScale, QColor col)
	{
		m_infos =inf;
		m_position = pos;
		m_sizePolicy= size;
		m_zScale = zScale;
		m_color = col;
	}

	InfoTooltip( QString inf,QVector3D pos,int size, int zScale, QColor col,QFont font)
		{
			m_infos =inf;
			m_position = pos;
			m_sizePolicy= size;
			m_zScale = zScale;
			m_color = col;
			m_font = font;
		}

	QVector3D position()
	{
		return m_position;
	}

	QString getName()
	{
		return  m_infos;
	}

	float getZScale()
	{
		return m_zScale;
	}

	int getSizePolicy()
	{
		return m_sizePolicy;
	}

	void setSizePolicy(int size)
	{
		m_sizePolicy = size;
	}

	bool getBold()
	{
		return m_bold;
	}

	bool getItalic()
	{
		return m_italic;
	}

	QString getPolicy()
	{
		return m_policy;
	}

	QColor getColor()
	{
		return m_color;
	}

	QFont getFont()
	{
		return m_font;
	}


	void setColor(QColor col)
	{
		m_color = col;
	}
	void setFont(QFont ft)
	{
		m_font = ft;
	}

private:

	QString m_infos;
	QVector3D m_position;
	int m_sizePolicy;
	float m_zScale;
	bool m_bold;
	bool m_italic;
	QString m_policy;
	QColor m_color;
	QFont m_font;

};


class widgetName: public QDialog
{
	Q_OBJECT
public:
	widgetName(QWidget* parent):QDialog(parent)
	{
		setWindowTitle("Tooltip 3D");
		setMinimumWidth(300);
		setMinimumHeight(200);
		setMaximumHeight(200);
		setModal(true);

		QWidget* wid= new QWidget();
		QVBoxLayout* LayoutG = new QVBoxLayout();
		LayoutG->setSpacing(20);

		QLabel* label1= new QLabel("Name",parent);
		m_editName = new QTextEdit(parent);

	/*	m_comboSize= new QComboBox(parent);
		m_comboSize->addItem("12");m_comboSize->addItem("14");m_comboSize->addItem("16");
		m_comboSize->addItem("18");m_comboSize->addItem("20");m_comboSize->addItem("22");
		m_comboSize->addItem("24");m_comboSize->addItem("26");m_comboSize->addItem("28");



		connect(m_comboSize, SIGNAL(currentIndexChanged(int)), this, SLOT(setSize(int)));
		m_comboSize->setCurrentIndex(1);
*/



		QPushButton* m_paramButton = new QPushButton("Font",parent);
		m_paramButton->setIcon(QIcon(QString("slicer/icons/graphic_tools/text.png")));//":/slicer/icons/property.svg")));
		m_paramButton->setIconSize(QSize(36, 36));
		m_paramButton->setToolTip("Font settings");

		connect(m_paramButton, &QPushButton::clicked, this, &widgetName::showFont);

		QPushButton* m_paramButton2 = new QPushButton("Color",parent);
	//	m_paramButton2->setIcon(QIcon(QString("slicer/icons/graphic_tools/text.png")));//":/slicer/icons/property.svg")));
	//	m_paramButton2->setIconSize(QSize(36, 36));
		m_paramButton2->setToolTip("Font color");

		connect(m_paramButton2, &QPushButton::clicked, this, &widgetName::showColor);


		LayoutG->addWidget(label1);
		LayoutG->addWidget(m_paramButton);
		LayoutG->addWidget(m_paramButton2);

		wid->setLayout(LayoutG);


		QVBoxLayout* Layout = new QVBoxLayout();

		QWidget* w= new QWidget();
		QHBoxLayout* LayoutH = new QHBoxLayout();
		LayoutH->addWidget(wid);
		LayoutH->addWidget(m_editName);
		w->setLayout(LayoutH);



		QDialogButtonBox *buttonBox = new QDialogButtonBox(
					QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

		Layout->addWidget(w);
		Layout->addWidget(buttonBox);

		setLayout(Layout);



		connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
		connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

		connect(m_editName, SIGNAL(textChanged()), this, SLOT(setName()));

		setFocus();



	}

	QString getName()
	{
		return m_name;
	}

	int getSize()
	{
		return m_size;
	}
	QFont getFont()
	{
		return m_font;
	}

	QColor getColor()
	{
		return m_color;
	}

public slots:
	void setName()
	{
		m_name = m_editName->toPlainText();
	}

	void setSize(int index)
	{
		m_size = m_comboSize->currentText().toInt();
	}

	void showFont()
	{
		bool ok;
		m_font = QFontDialog::getFont(
		                &ok, QFont("Helvetica [Cronyx]", 10), this);
		if (ok) {
		    // the user clicked OK and font is set to the font the user selected
		} else {
		    // the user canceled the dialog; font is set to the initial
		    // value, in this case Helvetica [Cronyx], 10
		}
	}

	void showColor()
	{
		 QColorDialog dialog;
		dialog.setCurrentColor (m_color);
		dialog.setOption (QColorDialog::DontUseNativeDialog);

		/* Get new color */
		if (dialog.exec() == QColorDialog::Accepted)
			m_color =dialog.currentColor() ;//QVariant (dialog.currentColor()).toString();
	}

private:
	QString m_name;
	QTextEdit* m_editName;
	QComboBox* m_comboSize;
	QFont m_font;
	int m_size=14;
	QColor m_color=Qt::white;



};


class WidgetShorcut: public QDialog
{
public:
	WidgetShorcut(QWidget* parent):QDialog(parent)
	{
		setWindowTitle("Shortcut help");
		setMinimumWidth(460);
		setMinimumHeight(300);
		setModal(false);
		QLabel* label1= new QLabel(" <b>ShortCut</b>",parent);
		QLabel* label2= new QLabel(" <b>Description</b>",parent);
		QLabel* label3= new QLabel(" <b>Mode</b>",parent);

		QLabel* labelshortcut1= new QLabel("Down arrow",parent);
		QLabel* labeldesc1= new QLabel("Translate camera Z down",parent);
		QLabel* labelmode1= new QLabel("All",parent);

		QLabel* labelshortcut2= new QLabel("Up arrow",parent);
		QLabel* labeldesc2= new QLabel("Translate camera Z up",parent);
		QLabel* labelmode2= new QLabel("All",parent);

		QLabel* labelshortcut3= new QLabel("Mouse left button",parent);
		QLabel* labeldesc3= new QLabel("Moving camera to target",parent);
		QLabel* labelmode3= new QLabel("All",parent);

		QLabel* labelshortcut4= new QLabel("Mouse wheel button",parent);
		QLabel* labeldesc4 = new QLabel("Zoom camera",parent);
		QLabel* labelmode4 = new QLabel("Standard",parent);

		QLabel* labelshortcut5= new QLabel("Mouse left +move",parent);
		QLabel* labeldesc5= new QLabel("Orbital rotation around the target",parent);
		QLabel* labelmode5= new QLabel("Standard",parent);

		QLabel* labelshortcut6= new QLabel("Mouse wheel button",parent);
		QLabel* labeldesc6 = new QLabel("Speed move",parent);
		QLabel* labelmode6 = new QLabel("Helico",parent);

		QLabel* labelshortcut7= new QLabel("Mouse right +move",parent);
		QLabel* labeldesc7= new QLabel("Rotation camera target ",parent);
		QLabel* labelmode7= new QLabel("Helico",parent);

		QLabel* labelshortcut10= new QLabel("Mouse right +move",parent);
		QLabel* labeldesc10= new QLabel("Translation Lateral camera",parent);
		QLabel* labelmode10= new QLabel("Standard",parent);

		QLabel* labelshortcut9= new QLabel("Mouse right+ T",parent);
		QLabel* labeldesc9= new QLabel("Create tooltip ",parent);
		QLabel* labelmode9= new QLabel("All",parent);

		QLabel* labelshortcut8= new QLabel("touch V",parent);
		QLabel* labeldesc8= new QLabel("Add point in path 3D ",parent);
		QLabel* labelmode8= new QLabel("Animation",parent);

		QGridLayout *layout = new QGridLayout();
		layout->addWidget(label1, 0, 0, 1, 1);
		layout->addWidget(label2, 0, 1, 1, 2);
		layout->addWidget(label3, 0, 3, 1, 1);

		QFrame* line = new QFrame();
		line->setFrameShape(QFrame::HLine);
		line->setFrameShadow(QFrame::Sunken);
		layout->addWidget(line,1,0,1,4);

		layout->addWidget(labelshortcut1, 2, 0, 1, 1);
		layout->addWidget(labeldesc1, 2, 1, 1, 2);
		layout->addWidget(labelmode1, 2, 3, 1, 1);

		layout->addWidget(labelshortcut2, 3, 0, 1, 1);
		layout->addWidget(labeldesc2, 3, 1, 1, 2);
		layout->addWidget(labelmode2, 3, 3, 1, 1);

		layout->addWidget(labelshortcut3, 4, 0, 1, 1);
		layout->addWidget(labeldesc3, 4, 1, 1, 2);
		layout->addWidget(labelmode3, 4, 3, 1, 1);

		layout->addWidget(labelshortcut4, 5, 0, 1, 1);
		layout->addWidget(labeldesc4, 5, 1, 1, 2);
		layout->addWidget(labelmode4, 5, 3, 1, 1);

		layout->addWidget(labelshortcut5, 6, 0, 1, 1);
		layout->addWidget(labeldesc5, 6, 1, 1, 2);
		layout->addWidget(labelmode5, 6, 3, 1, 1);

		layout->addWidget(labelshortcut6, 7, 0, 1, 1);
		layout->addWidget(labeldesc6, 7, 1, 1, 2);
		layout->addWidget(labelmode6, 7, 3, 1, 1);

		layout->addWidget(labelshortcut7, 8, 0, 1, 1);
		layout->addWidget(labeldesc7, 8, 1, 1, 2);
		layout->addWidget(labelmode7, 8, 3, 1, 1);

		layout->addWidget(labelshortcut10, 9, 0, 1, 1);
		layout->addWidget(labeldesc10, 9, 1, 1, 2);
		layout->addWidget(labelmode10, 9, 3, 1, 1);

		layout->addWidget(labelshortcut9, 10, 0, 1, 1);
		layout->addWidget(labeldesc9, 10, 1, 1, 2);
		layout->addWidget(labelmode9, 10, 3, 1, 1);

		layout->addWidget(labelshortcut8, 11, 0, 1, 1);
		layout->addWidget(labeldesc8, 11, 1, 1, 2);
		layout->addWidget(labelmode8,11, 3, 1, 1);

		 setLayout(layout);

	}

};

class ViewQt3D: public AbstractInnerView, public IDepthView {
Q_OBJECT
public:

	ViewQt3D(bool restictToMonoTypeSplit,QString uniqueName);
	virtual ~ViewQt3D();

	void showRep(AbstractGraphicRep *rep) override;
	void hideRep(AbstractGraphicRep *rep) override;

	void resetZoom() override;

	const QMatrix4x4& sceneTransform() const;
	QRect3D worldBounds();

	const QMatrix4x4& sceneTransformInverse() const;

	float zScale() const;

	void setZScale(double zScale);
	void setPositionCam(QVector3D pos);
	void setViewCenterCam(QVector3D center);
	void setUpVectorCam(QVector3D up);

	bool isCameraSyncActive() const;

	QVector3D positionCam() const;
	QVector3D targetCam() const;

	bool getAnimRunning() const;

	//bool event(QEvent *event)override;

	void showHelico( bool);


	void keyPressEvent(QKeyEvent *event) override;
	void keyReleaseEvent(QKeyEvent *event) override;

//	void enterEvent(QEvent* event);
//	void leaveEvent(QEvent* event);

	void importNurbs(IsoSurfaceBuffer surface,QString nameNurbs);
	void importNurbsObj(QString path, QString nameNurbs);
	 Q_INVOKABLE void hideWell();
	 Q_INVOKABLE void deselectWell();


	 void refreshWellTooltip(QString name);
	 void refreshPickTooltip(QString name);



	void setVitesse(float v){ m_vitesse =v;}
	void addVitesse(float v){m_vitesse +=v;}
	float vitesse(){return m_vitesse;}


	float speedUpDown(){return m_speedUpDown;}

	void setInfosVisible(bool);
	void setGizmoVisible(bool);
	void setSpeedUpDown(float);
	void setSpeedHelico(float );
	void setSpeedRotHelico(float );

	void setSimplificationSurface(int value)
	{
		m_simplifySurface = value;
		emit signalSimplifySurface(m_simplifySurface);
	}

	void setWireframeWell(bool value )
	{
		m_wireframeWell = value;
		emit signalWireframeWell(m_wireframeWell);
	}

	void setShowNormalsWell(bool value )
	{
		m_showNormalsWell = value;
		emit signalShowNormalsWell(m_showNormalsWell);
	}

	void setSimplificationWell(double value )
	{
		m_simplifyWell = value;
		emit signalSimplifyWell(value);
	}

	void setSimplificationLogs(int value )
	{
		m_simplifyLogs = value;
		emit signalSimplifyLogs(value);
	}

	void setThicknessPick(int value)
	{
		if(m_thicknessPick != value)
		{
			m_thicknessPick = value;

			emit signalThicknessPick(m_thicknessPick);
		}
	}

	void setThicknessLog(int value)
	{
		if(m_thicknessLog != value)
		{
			m_thicknessLog = value;
			emit signalThicknessLog(m_thicknessLog);
		}
	}

	void setColorLog(QColor value)
	{
		if(m_colorLog != value)
		{
			m_colorLog = value;
			emit signalColorLog(m_colorLog);
		}
	}
	void setColorWell(QColor value)
	{
		if(m_colorWell != value)
		{
			m_colorWell = value;
			emit signalColorWell(m_colorWell);
		}
	}

	void setColorSelectedWell(QColor value)
	{
		if(m_colorSelectedWell != value)
		{
			m_colorSelectedWell = value;
			emit signalColorSelectedWell(m_colorSelectedWell);
		}
	}

	void setDiameterWell(int value)
	{
		if(m_diameterWell != value)
		{
			m_diameterWell = value;
			emit signalDiameterWell(m_diameterWell);
		}
	}

	void setDiameterPick(int value)
	{
		if(m_diameterPick != value)
		{
			m_diameterPick = value;
			emit signalDiameterPick(m_diameterPick);
		}
	}

	void setSpeedAnim( int value )
	{
		if(m_speedAnim != value)
		{
			m_speedAnim = value;
			emit signalSpeedAnim(m_speedAnim);
		}
	}

	void setAltitudeAnim( int value )
	{
		if(m_altitudeAnim != value)
		{
			m_altitudeAnim = value;
			emit signalAltitudeAnim(m_altitudeAnim);
		}
	}

	void sendAltitudeAnim()
	{
		emit signalAltitudeAnim(m_altitudeAnim);
	}

	void sendSpeedAnim()
	{
		emit signalSpeedAnim(m_speedAnim);
	}

	void sendColorWell()
		{
			emit signalColorWell(m_colorWell);
		}

	void sendColorSelectedWell()
	{
		emit signalColorSelectedWell(m_colorSelectedWell);
	}


	void sendDiameterWell()
		{
			emit signalDiameterWell(m_diameterWell);
		}

	void sendDiameterPick()
	{
		emit signalDiameterPick(m_diameterPick);
	}

	void sendThicknessPick()
	{

		emit signalThicknessPick(m_thicknessPick);
	}

	void sendSimplifyWell()
	{
		emit signalSimplifyWell(m_simplifyWell);
	}
	void sendWireframeWell()
	{
		emit signalWireframeWell(m_wireframeWell);
	}

	void sendShowNormalsWell()
	{
		emit signalShowNormalsWell(m_showNormalsWell);
	}

	void sendThicknessLog()
	{
		emit signalThicknessLog(m_thicknessLog);
	}
	void sendColorLog()
	{
		emit signalColorLog(m_colorLog);
	}

	QColor getLogColor()
	{
		return m_colorLog;
	}

//	Qt3DRender::QLayer* getLayerTransparent() const;
//	Qt3DRender::QLayer* getLayerOpaque() const;


	///gestion des randomView3d
	void createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture,QVector2D range, RandomLineView* random,GraphEditor_LineShape* line,QVector3D position,
			QVector3D normal);//, float width);
void createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture
			,QVector2D range, RandomLineView* random,GraphEditor_LineShape* line);
	void createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> listepoints,QVector<CudaImageTexture*> cudaTextures
				,QVector<QVector2D> ranges, RandomLineView* random,GraphEditor_LineShape* line);
	void createRandomView(bool isOrtho,QString nameView, QVector<QVector3D> ,CudaImageTexture* cudaTexture,QVector2D range,RandomLineView* random);
	void updateRandomView(QString nameView,QVector<QVector3D> listepoints,CudaImageTexture* cudaTexture,QVector2D range, bool followCam, float distance, float altitude, float inclinaison,int width,bool withTangent= false);
	void updateWidthRandomView(QString nameView,QVector<QVector3D> listepts , float width);

	void selectRandomView(int index);
	void selectRandomView(QString nameView);
	void deleteRandomView(QString nameView);
	int getIndexRandomView(QString nameView);

	int destroyRandomView(RandomLineView*);
	void deleteCurrentRandomView(RandomView3D*  rand3d);

	int findRandom3d(RandomLineView* );

	QVector2D getPos2D(QVector3D,float);




	InfoTooltip* findTooltip(QString s)
	{
		for(int i=0;i<m_listeTooltips.count();i++)
		{
			if(m_listeTooltips[i]->getName() ==s)
				return m_listeTooltips[i];
		}
		return nullptr;
	}

	void createTooltip(QString nom, QVector3D pos, int sizePolicy,float zScale,QString family,bool bold, bool italic,QColor color);
	void destroyAllTooltip();
	QVector<InfoTooltip*> getAllTooltip();





	void exportNurbsObj(QString nom);


	virtual const MtLengthUnit* depthLengthUnit() const override;



public slots:
	void onZScaleChanged(int val);
	void zScaleChanged(double val);
	void positionCameraChanged(QVector3D pos);
	void viewCenterCameraChanged(QVector3D center);
	void upVectorCameraChanged(QVector3D up);
	void etatViewChanged(bool b);


	void distanceTargetChanged(float distance,QVector3D target );

	void changeDistanceForcing(float d);

	void showHelp();
	//void showProperty();
	void downCamera();
	void upCamera();
	void stopDownUpCamera();

	void showTooltipWell(const IToolTipProvider* tooltipProvider, QString,int,int,QVector3D );

	void showTooltipPick(const IToolTipProvider* tooltipProvider, QString,int , int ,QVector3D );
	void hideTooltipWell();

	void selectWell(WellBoreLayer3D*);


	void moveLineVert(float posX, float posZ);

	void showLineVert(bool b);
	void recordPath3d();
	void playPath3d();

	void stopPath3d();

	void show3DTools();

	void deleteRandom(RandomView3D*);

	Qt3DCore::QEntity* getRoot()
	{
		return m_root;
	}

	float getAltitude(QVector3D pos)
	{
		CameraController* cameraCtrl = dynamic_cast<CameraController*>(m_controler);
		if (cameraCtrl != nullptr)
		{
			return cameraCtrl->computeHeight( pos);
		}
		return 0.0f;
	}

	void showCamTooltip(int);

	void setSizePolicy(int, QString,int);
	void setColorTooltip(int,QString, QColor);
	void setFontTooltip(int index ,QString name ,QFont font);

	//gestion des nurbs3d
	void createNurbs(QVector<QVector3D> ,bool withTangent,IsoSurfaceBuffer surface,GraphEditor_ListBezierPath* path , QString nameNurbs="",QColor col= Qt::yellow);
	void updateNurbs(QVector<QVector3D> ,bool withTangent,QColor col);
	void updateNurbs(GraphEditor_ListBezierPath* path,QColor col);


	void createNurbsSimple( QString nameNurbs,IsoSurfaceBuffer buffer);

	void selectNurbs(int index);
	void selectNurbs(QString);

	void destroyNurbs(QString name);
	void deleteNurbs(QString);
	void deleteGeneratriceNurbs(QString);
	void createSection(QVector<QVector3D> ,int index=-1,bool isopen = true,bool withTangent =false);
	void createSection(QVector<PointCtrl> listeCtrls,QVector<QVector3D> listepoints,int index,bool isopen,bool withTangent,QPointF cross,QVector<QVector3D>  listeCtrl3D,QVector<QVector3D>  listeTangent3D,QVector3D cross3D);

	void createSection(GraphEditor_ListBezierPath* path,RandomTransformation* transfo,int index);


	void createNewXSection(float);
	void createNewXSectionClone(float);
	void setSliderXsection(float);
	void setSliderXsection(float pos,QVector3D position,QPointF normal);

	void setPrecisionNurbs(int);
	void setInterpolationNurbs(bool);
	void setWireframeNurbs(bool);

	void receiveNurbsY(QVector3D, QVector3D);
	void receiveCurveData(std::vector<QVector3D> listePts,bool isopen);
	void receiveCurveData(QVector<PointCtrl> listePts,bool isopen,QPointF);
	void receiveCurveData2(QVector<QVector3D> listePts,QVector<QVector3D> globalTangente3D,bool isopen,QPointF cross,QString);
	void receiveCurveDataOpt(GraphEditor_ListBezierPath* path);



	void setColorNurbs(QColor col)
	{
		m_nurbsManager->setColorNurbs(col);
	}

	void setColorNurbs(QString name, QColor col)
		{
			m_nurbsManager->setColorNurbs(name, col);
		}

	void setColorDirectrice(QColor col)
	{
		m_nurbsManager->setColorDirectrice(col);
	}

	NurbsManager* getManagerNurbs()
	{
		return m_nurbsManager;
	}

	float getHeightBox()
	{
		return m_heightBox;
	}

	void toggleDepthUnit();
	virtual void setDepthLengthUnit(const MtLengthUnit* depthLengthUnit) override;


protected slots:

	void externalMouseMoved(MouseTrackingEvent *event) override {
	/*	float YY =0.0f;
		if(event->hasDepth()==true)
		{
			qDebug()<<"vent->depth :"<<event->depth();
			YY= event->depth()*0.5f;
		}
		qDebug()<<m_ydepart<<" , m_heightBox :"<<m_heightBox;*/
		if( m_line!= nullptr)
		{
			showLineVert(true);
			QVector3D pos2 = m_sceneTransform* QVector3D(event->worldX(),0.0f,event->worldY());
			pos2.setY(0.0f);
			//qDebug()<<" pos2 :"<<pos2;
			m_lineTransfo->setTranslation(pos2);

		}
	}
	void onQMLReady(QQuickView::Status status);
	void toggleSyncCamera();
	void toggleFlyCamera();

	void onAnimationFinished();
	void onAnimationStopped();

	void addNewInfosCam(QVector3D , QVector3D);

	void resetCamera();
	void refreshTooltip();

	void setAnimationCamera(int, QVector3D);
	void setPositionTooltip(QVector3D);

	void deleteTooltip(QString);



signals:

	void refreshOrthoFromBezier();
	void zScaleChangedSignal(double zScale);
	void positionCamChangedSignal(QVector3D pos);
	void viewCenterCamChangedSignal(QVector3D center);
	void upVectorCamChangedSignal(QVector3D up);

	void showHelico2D(bool b);

	void distanceTargetChangedSignal(float );


	void signalSimplifyLogs(int );
	void signalSimplifyWell(double );
	void signalWireframeWell(bool);
	void signalShowNormalsWell(bool);
	void signalSimplifySurface(int);

	void signalThicknessPick(int);
	void signalDiameterPick(int);
	void signalThicknessLog(int);
	void signalColorLog(QColor);
	void signalColorWell(QColor);
	void signalColorSelectedWell(QColor);
	void signalDiameterWell(int);

	void signalSpeedAnim(int);
	void signalAltitudeAnim(int);


	void signalShowTools(bool);

	void sendCoefNurbsY(float);
	void sendCoefNurbsXYZ(QVector3D);
	void sendCurveChanged(std::vector<QVector3D>,bool);
	void sendCurveChanged(QVector<PointCtrl>,bool,QPointF);
	void sendCurveChangedTangent(QVector<QVector3D>,QVector<QVector3D>,bool,QPointF,QString nameNurbs);
	void sendCurveChangedTangentOpt(GraphEditor_ListBezierPath* path);
	void sendNurbsName(QString);
	void receiveOrtho(QVector3D,QVector3D);

	void sendDeleteRandom3D(RandomView3D *);

	void sendDestroyTooltip(QString);
	void sendNewTooltip(QString);
	void addTooltip(int positionX,int positionY ,QString mess,int size,QString ,bool,bool,QColor);
	void updateTooltip(int index, int posx,int posy);
	void sendUpdateTooltip(int index,int size);
	void sendColorTooltip(int index,QColor color);
	void sendFontTooltip(int index,QString police,int size,bool italic,bool bold);

	void removeAllTooltip();



protected:
	bool absoluteWorldToViewWorld(MouseTrackingEvent &event) override {
		return true;
	}
	bool viewWorldToAbsoluteWorld(MouseTrackingEvent &event) override {
		return true;
	}

	virtual bool updateWorldExtent(const QRect3D &worldExtent);
	void updateControler();
	void updateSyncButton();
protected:

	QQuickView *m_quickview = nullptr;
    //QQuickView *m_camerhudview;

	Qt3DCore::QEntity *m_root = nullptr;
	Qt3DCore::QEntity *m_controler = nullptr;

	Qt3DCore::QEntity *m_gizmo = nullptr;

	Qt3DRessource *m_GPURes = nullptr;
	QToolButton* m_syncButton = nullptr;

	QToolButton* m_flyButton = nullptr;
	QSpinBox* m_spinZScale = nullptr;

	QToolButton* m_recordButton= nullptr;
	//QToolButton* m_playButton=nullptr;

	bool m_recordPathMode = false;
	bool m_playPathMode = false;

	QObject* m_infos3d = nullptr;
	QObject* mToolTip = nullptr;
	QObject* mToolTip3D = nullptr;
	QObject* mLineTooltip = nullptr;

	QObject* m_textToolTip3D= nullptr;
	QObject* m_statustooltip= nullptr;
	QObject* m_uwitooltip= nullptr;
	QObject* m_domaintooltip= nullptr;
	QObject* m_elevtooltip= nullptr;
	QObject* m_datumtooltip= nullptr;
	QObject* m_velocitytooltip= nullptr;
	QObject* m_ihstooltip= nullptr;
	QObject* m_deselecttooltip=nullptr;
	QObject* m_datetooltip=nullptr;

	WellBoreLayer3D* m_lastSelected;
	Qt3DCore::QEntity* m_line = nullptr;
	Qt3DCore::QTransform* m_lineTransfo;
	Qt3DExtras::QCylinderMesh* m_cylMesh =nullptr;

	//Qt3DRender::QLayer*m_layerTransparent=nullptr;
//	Qt3DRender::QLayer *m_layerOpaque=nullptr;

	QVector3D m_lastPos3DWeel;

	QRect3D m_worldBounds;
	bool m_wordBoundsInitialized;

	float m_currentScale;

	QMatrix4x4 m_sceneTransform;
	QMatrix4x4 m_sceneTransformInverse;
	SampleUnit m_sectionType;
	const MtLengthUnit* m_depthLengthUnit;

	bool m_qmlLoaded;
	QQueue<AbstractGraphicRep*> m_waitingReps; // reps that await loading

	bool m_syncCamera = true;
	bool m_flyCamera = false;

	QPropertyAnimation* m_animation;
	QPropertyAnimation* m_animationCenter;
	bool m_animRunning;

	Qt3DCore::QTransform* m_transfoRoot = nullptr;
	Qt3DCore::QTransform* m_transfoRootFils = nullptr;

	Qt3DCore::QEntity* m_pointer;
	Qt3DCore::QEntity* m_coneEntity;
	Qt3DExtras::QConeMesh* m_coneMesh;
	Qt3DCore::QTransform* m_coneTransfo;

	float m_vitesse = 0.5f;
	float m_speedUpDown;
	QToolButton* m_upButton;
	//QToolButton* m_depthUnitToggle;

	WidgetShorcut* m_shortcut;


	float m_ydepart=0.0f;
	float m_heightBox=0.0f;

	bool m_visibleGizmo3d;
	bool m_visibleInfos3d;

private:
	int m_diameterWell = 30;
	double m_simplifyWell = 2.0f;
	int m_simplifyLogs =1;
	int m_simplifySurface =10;
	bool m_wireframeWell= false;
	bool m_showNormalsWell = false;
	int m_diameterPick = 50;
	int m_thicknessPick = 15;
	int m_thicknessLog = 3;
	QColor m_colorLog = Qt::green;
	QColor m_colorWell = Qt::yellow;
	QColor m_colorSelectedWell = Qt::cyan;
	int m_altitudeAnim=400;
	int m_speedAnim= 200;

	bool m_followCam = false;
	float m_distanceCam = 200.0f;
	//float m_altitudeCam = 0.0f;
	 float m_inclinaisonCam = 0.0f;
	 float m_targetY =0.0f;

	QVector<InfoTooltip*> m_listeTooltips;

		QVector<RandomView3D*> m_randomViews;
		int m_currentIndexRandom =-1;

		int m_lastSelectedViews = -1;

		float m_coefGlobal = 1.0f;

	// can either be a well or a pick layer
	// const : because it should not be modified but used to access tooltip information
	const IToolTipProvider* m_currentTooltipProvider = nullptr;
	bool m_tooltipProviderIsWell = true;

public:
	Qt3DRender::QCamera *m_camera = nullptr;
	WidgetPath3d* widgetpath = nullptr;


	//Manager* m_managerNurbs;

	NurbsManager* m_nurbsManager= nullptr;

	WorkingSetManager* m_working = nullptr;
	bool m_tooltipActif = false;


	QPointF m_currentCross;

	QWidget* m_horizontalToolWidget;
	QHBoxLayout* m_horizontalToolLayout;


	//Qt3DRender::QMaterial* m_materialRandom;




};

#endif
