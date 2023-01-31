#ifndef NURBSWIDGET_H
#define NURBSWIDGET_H

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
#include "GraphicSceneEditor.h"


class Abstract2DInnerView;
class GraphicSceneEditor;
class GraphEditor_Path;


class widgetNameForSave: public QDialog
{
	Q_OBJECT
public:
	widgetNameForSave(QString name, QWidget* parent):QDialog(parent)
	{
		setWindowTitle(name);
		setMinimumWidth(300);
		setMinimumHeight(100);
		setMaximumHeight(100);
		setModal(true);


		QLabel* label1= new QLabel("Name",parent);
		m_editName = new QTextEdit(parent);
		m_editName->setFocus();

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


};


class NurbsWidget: public QDialog
{
	Q_OBJECT

public:


	 	 static void setView2D(Abstract2DInnerView* view2d);
		 static void setView3D(ViewQt3D* view3d);
		 static void setInlineView(Abstract2DInnerView* viewInline);

		 static void showWidget();
		 static void closeWidget();

		 static void addNurbs(QString path,QString name);

		 static void removeNurbs(QString path,QString name);
		 static void editerNurbs(QString path,QString name);

		 static int getNbNurbs();
		 static QString getPath(int);
		 static QString getName(int);

		 static void addCombo(QString name);
		 static void clearCombo();


		 static IsoSurfaceBuffer getHorizonBufferValid();
		 GraphEditor_Path* getDirectriceBezier();

		 void setModeEditable(bool edit);

		 static void addView3d(ViewQt3D* view3d);
		 static void removeView3d(ViewQt3D* view3d);

		 static QColor getCurrentColor();
		 static void setColor(QString s, QColor col);
		 static void saveColor(QString path, QColor col);
		 static QString newNurbsSimple();

	public slots:
		void editerNurbs();
		 void generateDirectrice();
		 void addGeneratrice();
		 void exportNurbs();
		 void newNurbs();
		// void importNurbs();

		 void activerDirectrice();
		 void activerGeneratrice();

		 void setSelected(int index);


		 void supprimerDirectrice();
		 void supprimerGeneratrice();

	//	 void moveCrossSection( int value);


	/*	 void receivePointsNurbs(QVector<QPointF>  listepoints, bool withTangent);
		void updatePointsNurbs(QVector<QPointF>  listepoints, bool withTangent);
		void receiveCrossPoints(QVector<QPointF>  listepoints,bool isopen = true);
		void receiveCrossPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen);

*/

		 void receiveCrossPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen,QPointF cross);

		void setSelectNurbs(QString);
		void setDeleteNurbs(QString);

		void receiveColorNurbs(QColor,QColor,int,bool,int);

		 void receiveNameNurbs(QString name);
		 void nurbsSelectedChanged(int index);
		void setNurbsPrecision(int);

		// void setDirectriceColor();
			void setNurbsColor();



private:

		NurbsWidget(QWidget* parent= nullptr);
		virtual ~NurbsWidget();
		static NurbsWidget* m_pInstance;

		QComboBox *m_comboNurbs;
		QToolButton* validerGeneratrice;
		QToolButton* validerDirectrice;

		QToolButton* buttonSupprGeneratrice;
		QToolButton* buttonSupprDirectrice;

		//QPushButton* m_buttonDirectrice;
		QPushButton* m_buttonNurbs;
		QSlider *m_sliderPrecision;
		QLabel* labelQuality;

		QToolButton *m_editButton;
	//	QLabel* label2;
		QLabel* label3;
		QLabel* labelDirectrice;
		QLabel* labelGeneratrice;
		QPushButton* saveNurbs;
	//	QLabel* m_labelLayer;


		//QColor m_directriceColor = QColor(255,255,0,255);
		static QColor m_nurbsColor;// = QColor(180,180,180,255);

		bool m_directriceOK = false;
		bool m_generatriceOK = false;

		static GraphicSceneEditor* m_sceneEditorRGB;
		static GraphicSceneEditor* m_sceneEditorSpectrum;
		//static GraphicSceneEditor* m_sceneEditorGCC;
		static GraphicSceneEditor* m_sceneEditor2;

		 //3D view
		static  ViewQt3D* m_view3D;

		static QList<ViewQt3D*> m_listeView3D;

		 //2D view
		static QList<QPointer<Abstract2DInnerView>> m_listeView2D;

		static Abstract2DInnerView* m_view2D;

	//	static Abstract2DInnerView* m_view2Dmean;
	//	static Abstract2DInnerView* m_view2DGCC;

		 //inlineView
		static Abstract2DInnerView* m_viewInline;

		float m_coefPosition=0.0f;

		static QVector<QString> m_listeLoadNurbs;
		static QVector<QString> m_listeNameNurbs;

		static bool first;

		int m_indexView = -1;





};

#endif
