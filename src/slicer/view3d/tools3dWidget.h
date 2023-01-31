#ifndef TOOLS3DWIDGET_H
#define TOOLS3DWIDGET_H

#include <QDialog>
#include <QGroupBox>
#include <QLineEdit>
#include <QGridLayout>
#include <QToolButton>
#include <QDial>
#include <QTimer>
#include "viewqt3d.h"
#include "abstract2Dinnerview.h"
#include "randomrep.h"
#include <QVector>
#include <QPointF>
#include "cudaimagepaletteholder.h"


class Abstract2DInnerView;

class GraphEditor_LineShape;
class GraphEditor_Path;



const QString Tools3dWidgetTitle = "3D Tools widget";


/*
class widgetNameForSave: public QDialog
{
	Q_OBJECT
public:
	widgetNameForSave(QWidget* parent):QDialog(parent)
	{
		setWindowTitle("Nurbs ");
		setMinimumWidth(300);
		setMinimumHeight(100);
		setMaximumHeight(100);
		setModal(true);


		QLabel* label1= new QLabel("Name",parent);
		m_editName = new QTextEdit(parent);

		QVBoxLayout* Layout = new QVBoxLayout();

		QWidget* w= new QWidget();
		QHBoxLayout* LayoutH = new QHBoxLayout();
		LayoutH->addWidget(label1);
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

public slots:
	void setName()
	{
		m_name = m_editName->toPlainText();
	}


private:
	QString m_name;
	QTextEdit* m_editName;


};*/

class Tools3dWidget: public QDialog
{
	 Q_OBJECT
public:
	 Tools3dWidget( QWidget* parent);

	 void setView2D(Abstract2DInnerView* view2d);
	 void setView3D(ViewQt3D* view3d);
	 void setInlineView(Abstract2DInnerView* viewInline);

	 void removeView2D(Abstract2DInnerView* view2d);
	 void removeView3D(ViewQt3D* view3d);

	 void AddPointsNurbs(QVector3D pos);

	 void moveSectionPosition(QPointF pos2D,QPointF nor2D);
	 signals:
	 void sendDeleteTooltip(QString);
	 void showTooltip(int);
	 void updateSizePolicy(int,QString, int);
	 void colorTooltipChanged(int,QString,QColor);
	 void fontTooltipChanged(int,QString,QFont);
	 void moveOrthoLine(QPointF,QPointF);

public slots:

	void currentViewSelected(Abstract2DInnerView* view2d);
	void nextKey(float);
	void selectRandomView();
	void deleteRandomView();
	void destroyRandomView(RandomLineView* );
	void deletedRandom3D(RandomView3D* random3d);
	void animationChanged(bool);

	void deleteGeneratrice(QString);

	void destroyRandomLineView();


	void updateOrthoSection(QPolygonF);
	void updateWidthOrtho(QPolygonF);
	void showRandomView(bool isOrtho,QVector<QPointF>  listepoints);

	void showRandomView(bool isOrtho,GraphEditor_LineShape* line, RandomLineView * randomOrtho,QString name);

	void receivePointsNurbs(QVector<QPointF>  listepoints, bool withTangent,GraphEditor_ListBezierPath* path ,QString nameNurbs,QColor col);
	void updatePointsNurbs(QVector<QPointF>  listepoints, bool withTangent,QColor);
	void updatePointsNurbs(GraphEditor_ListBezierPath* path,QColor col);
	void receiveCrossPoints(QVector<QPointF>  listepoints,bool isopen = true);
	void receiveCrossPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen,QPointF cross);
	void receiveCrossPoints(GraphEditor_ListBezierPath* path);

	void nurbsSelectedChanged(int index);

	void receiveNaneNurbs(QString name);
	void setSelectNurbs(QString);
	void setDeleteNurbs(QString);

	void receiveColorNurbs(QColor,QColor,int,bool,int);
	void setNurbsPrecision(int);
	void setLinearInterpolation(int);
	void setWireframe(int);

	 void moveCrossSection( int,int,QString );
	 void moveSection( );

	 void setIndexCurrentPts(int);

	 void addNewXSection();
	 void addNewXSectionClone();

	 void attachCamera(int);

	 void cameraFollow(int val);
	 void setDistance(int val);
	 void setAltitude(int val);
	 void setInclinaison(int val);
	 void setSpeed(int val);

	 void setChangedDistance(int val);
	 void resetChangedDistance();
	 void refreshChangedDistance();
	 void onTimeout();

	 void resetChangedAltitude();
	void refreshChangedAltitude();
	void onTimeoutAlt();

	 void receiveNewTooltip(QString);
	 void deleteTooltip();
	 void showCamTooltip();
	 void saveTooltip();
	 void loadTooltip();

	 void showFont();
	 void showColor();

	 void setSize(int);
	 void selectCombo(int index);

	 void updateOrthoFrom3D(QVector3D,QVector3D);

	 void saveNurbs();
	 void loadNurbs();
	 void exportNurbs();

	 void setDirectriceColor();
	 void setNurbsColor();

	 void refreshOrthoFromListBezier();

private:
	 void updateTexture(CudaImageTexture * texture,CUDAImagePaletteHolder *img );

	 bool isListBezierPath();

	 GraphEditor_Path* isListBezierPath(QString name);

	 QString getUniqueNameRandom();

	QVector<QVector3D> m_listePts3D;
	int m_indexCurrentPts = -1;

	 //3D view
	 ViewQt3D* m_view3D;

	 QVector<ViewQt3D* > listView3D;


	 //2D view
	 Abstract2DInnerView* m_view2D;
	 QVector<Abstract2DInnerView* > listView2D;
	 int m_indexView=-1;

	 //inlineView
	Abstract2DInnerView* m_viewInline;



	RandomRep* m_randomRep;
	int m_indexRandomRep;


	// UI
	QLineEdit* m_editline;
	QComboBox* m_comboViews;
	QComboBox *m_comboNurbs;
	QComboBox *m_comboTooltip;

	QComboBox * m_comboSize;
	QPushButton* m_buttonDirectrice;
	QPushButton* m_buttonNurbs;
	//QSlider *m_sliderAlt;
	QDial* m_moletteDist;
	QDial* m_moletteVitesse;
	QDial* m_moletteAltitude;
	QDial* m_moletteInclinaison;
	QColor m_directriceColor = QColor(255,255,0,255);
	QColor m_nurbsColor = QColor(0,0,255,255);

	float m_max;
	float m_coefPosition = 0.0f;

	bool m_cameraFollow=false;
	float m_distanceCam = 100.0f;
	float m_altitudeCam = 500.0f;
	float m_inclinaisonCam = 0.0f;


	QTimer* m_timerDist;
	QTimer* m_timerAlt;

	bool m_animationRunning = false;

	QFont m_fontTooltip;
	QColor m_colorTooltip = Qt::white;

	//QVector<InfoTooltip*> m_listeTooltip;

};

#endif
