#include "nurbswidget.h"
#include "randomlineview.h"
#include <QColorDialog>
#include "singlesectionview.h"
#include <QMessageBox>

NurbsWidget* NurbsWidget::m_pInstance = nullptr;

GraphicSceneEditor* NurbsWidget::m_sceneEditorSpectrum = nullptr;
GraphicSceneEditor* NurbsWidget::m_sceneEditorRGB = nullptr;
GraphicSceneEditor* NurbsWidget::m_sceneEditor2 = nullptr;

ViewQt3D* NurbsWidget::m_view3D = nullptr;
Abstract2DInnerView* NurbsWidget::m_view2D = nullptr;
//Abstract2DInnerView* NurbsWidget::m_view2DGCC = nullptr;
//Abstract2DInnerView* NurbsWidget::m_view2Dmean = nullptr;
Abstract2DInnerView*  NurbsWidget::m_viewInline = nullptr;

QVector<QString>  NurbsWidget::m_listeLoadNurbs;
QVector<QString>  NurbsWidget::m_listeNameNurbs;

QList<ViewQt3D*> NurbsWidget::m_listeView3D;
//QList<Abstract2DInnerView*> NurbsWidget::m_listeView2D;
QList<QPointer<Abstract2DInnerView>> NurbsWidget::m_listeView2D;
QColor NurbsWidget::m_nurbsColor = QColor(180,180,180,255);

 bool NurbsWidget::first=true;

NurbsWidget::NurbsWidget(QWidget* parent):QDialog(parent)
{


		setWindowTitle("Create Nurbs");
		setModal(false);
	//	setMinimumWidth(350);
		//setMinimumHeight(300);//300

		QScreen *screen = QGuiApplication::primaryScreen();
		QRect  screenGeometry = screen->geometry();

		int posX = screenGeometry.width() *0.5-500;
		//qDebug()<<" posX "<<posX <<" , width : "<<screenGeometry.width();
		setGeometry(posX,screenGeometry.height()*0.5-150,350,300);


		QGridLayout *layout = new QGridLayout();

		//les nurbs
		///	QGroupBox* boxNurbs = new QGroupBox("Nurbs");

			QLabel* labelnurbs= new QLabel("Nurbs");
			m_comboNurbs = new QComboBox();

			connect(m_comboNurbs, SIGNAL(currentIndexChanged(int)),this, SLOT(setSelected(int)));

			//QPushButton* newNurbs = new QPushButton("New");
			//connect(newNurbs, SIGNAL(clicked()),this, SLOT(newNurbs()));


			m_editButton = new QToolButton();
			m_editButton->setIcon(QIcon(QString(":/slicer/icons/graphic_tools/bezier.png")));
			m_editButton->setIconSize(QSize(20, 20));// :/slicer/icons/graphic_tools/bezier.png"
			m_editButton->setToolTip("Edit nurbs");
			m_editButton->setVisible(false);
			connect(m_editButton, SIGNAL(clicked()),this, SLOT(editerNurbs()));

			//QCheckBox* visiblecheckbox = new QCheckBox("visible");
			//visiblecheckbox->setCheckState(Qt::Checked);


			labelQuality= new QLabel("Precision");
			m_sliderPrecision = new QSlider(Qt::Horizontal);
			m_sliderPrecision->setMinimum(10);
			m_sliderPrecision->setMaximum(100);
			m_sliderPrecision->setValue(20);

			connect(m_sliderPrecision,SIGNAL(valueChanged(int)), this, SLOT(setNurbsPrecision(int)));
		//	connect(m_comboNurbs,SIGNAL(currentIndexChanged(int)),this,SLOT(nurbsSelectedChanged(int)));


		//	m_labelLayer = new QLabel("Layer");


		/*	label2= new QLabel("Directrice");
			m_buttonDirectrice = new QPushButton;
			QString namecolor2= "QPushButton {background-color: rgb("+QString::number(m_directriceColor.red())+","+QString::number(m_directriceColor.green())+","+QString::number(m_directriceColor.blue())+")}";
			m_buttonDirectrice->setStyleSheet(namecolor2);
			connect(m_buttonDirectrice,SIGNAL(clicked()),this,SLOT(setDirectriceColor()));*/




			label3= new QLabel("Nurbs");
			m_buttonNurbs = new QPushButton;
			//QColor color3(0,0,255,255);
			QString namecolor3= "QPushButton {background-color: rgb("+QString::number(m_nurbsColor.red())+","+QString::number(m_nurbsColor.green())+","+QString::number(m_nurbsColor.blue())+")}";
			m_buttonNurbs->setStyleSheet(namecolor3);
			connect(m_buttonNurbs,SIGNAL(clicked()),this,SLOT(setNurbsColor()));


			labelDirectrice= new QLabel("Pick on map (directrice)");
			labelGeneratrice= new QLabel("Pick on section (generatrice)");

			//QPushButton* validerDirectrice = new QPushButton("Valider");
			//QPushButton* validerGeneratrice = new QPushButton("Valider");

			int sizeIcon=24;
			if(m_pInstance)
			{
				m_pInstance->m_listeLoadNurbs.clear();
				m_pInstance->m_listeNameNurbs.clear();
			}

			validerDirectrice = new QToolButton();
			validerDirectrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
			validerDirectrice->setIconSize(QSize(sizeIcon, sizeIcon));
			validerDirectrice->setVisible(false);

			buttonSupprDirectrice = new QToolButton();
			buttonSupprDirectrice->setIcon(QIcon(":/slicer/icons/graphic_tools/delete.png"));// QStyle::SC_TitleBarContextHelpButton));
			buttonSupprDirectrice->setIconSize(QSize(sizeIcon, sizeIcon));
			buttonSupprDirectrice->setToolTip("Delete directrice");
			buttonSupprDirectrice->setVisible(false);


			QWidget* widgetButton1 = new QWidget(this);
			QHBoxLayout* lay1 = new QHBoxLayout();
			lay1->addWidget(validerDirectrice);
			lay1->addWidget(buttonSupprDirectrice);

			widgetButton1->setLayout(lay1);


			validerGeneratrice = new QToolButton();
			validerGeneratrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
			validerGeneratrice->setIconSize(QSize(sizeIcon, sizeIcon));

			validerGeneratrice->setVisible(false);

			buttonSupprGeneratrice = new QToolButton();
			buttonSupprGeneratrice->setIcon(QIcon(":/slicer/icons/graphic_tools/delete.png"));// QStyle::SC_TitleBarContextHelpButton));
			buttonSupprGeneratrice->setIconSize(QSize(sizeIcon, sizeIcon));
			buttonSupprGeneratrice->setToolTip("Delete generatrice");
			buttonSupprGeneratrice->setVisible(false);


			QWidget* widgetButton2 = new QWidget(this);
			QHBoxLayout* lay2 = new QHBoxLayout();
			lay2->addWidget(validerGeneratrice);
			lay2->addWidget(buttonSupprGeneratrice);

			widgetButton2->setLayout(lay2);


			connect(validerDirectrice, SIGNAL(clicked()),this, SLOT(generateDirectrice()));
			connect(validerGeneratrice, SIGNAL(clicked()),this, SLOT(addGeneratrice()));


			connect(buttonSupprDirectrice, SIGNAL(clicked()),this, SLOT(supprimerDirectrice()));
			connect(buttonSupprGeneratrice, SIGNAL(clicked()),this, SLOT(supprimerGeneratrice()));


		/*	QPushButton* addNurbs = new QPushButton("Import");
			connect(addNurbs, SIGNAL(clicked()),this, SLOT(importNurbs()));*/

			saveNurbs = new QPushButton("Save");
			connect(saveNurbs, SIGNAL(clicked()),this, SLOT(exportNurbs()));


			layout->addWidget(labelnurbs, 0, 0, 1, 1);
			layout->addWidget(m_comboNurbs, 0, 1, 1, 2);
			//layout->addWidget(newNurbs, 0, 3, 1, 1);
		//	layout->addWidget(loadNurbsButton, 0, 2, 1, 1);
			//layout->addWidget(saveNurbsButton, 0, 3, 1, 1);

			//layout->addWidget(label2, 1, 0, 1, 1);
			//layout->addWidget(m_buttonDirectrice, 1, 1, 1, 1);
			layout->addWidget(label3, 1, 0, 1, 1);
			layout->addWidget(m_buttonNurbs, 1, 2, 1, 1);
			layout->addWidget(labelQuality, 2, 0, 1, 1);
			layout->addWidget(m_sliderPrecision, 2, 1, 1, 1);
			///layout->addWidget(m_editButton, 2, 2, 1, 1);
		//	layout->addWidget(m_labelLayer, 2, 3, 1, 1);
			//layout->addWidget(visiblecheckbox, 2, 3, 1, 1);

			layout->addWidget(labelDirectrice, 3, 0, 1, 3);
			layout->addWidget(widgetButton1, 3, 3, 1, 1);

			//layout->addWidget(validerDirectrice, 3, 3, 1, 1);
			//layout->addWidget(buttonSupprDirectrice, 3, 4, 1, 1);
			layout->addWidget(labelGeneratrice, 4, 0, 1, 3);
			layout->addWidget(widgetButton2, 4, 3, 1, 1);
			//layout->addWidget(validerGeneratrice, 4, 3, 1, 1);
			//layout->addWidget(buttonSupprGeneratrice, 4, 4, 1, 1);

			//layout->addWidget(addNurbs, 5, 1, 1, 1);
			layout->addWidget(saveNurbs, 5, 3, 1, 1);

		setLayout(layout);
	}

NurbsWidget::~NurbsWidget()
{

	if(this== m_pInstance) m_pInstance = nullptr;
}

void NurbsWidget::showWidget()
{
	if (!m_pInstance)
		{
			m_pInstance = new NurbsWidget();
		}

		m_pInstance->show();
}

void NurbsWidget::closeWidget()
{
	if (m_pInstance)
		{
			m_pInstance->close();
			m_pInstance->deleteLater();
			m_pInstance=nullptr;
		}
}


void NurbsWidget::setSelected(int index)
{
	//qDebug()<<" set selected index"<<index;
	if(index>= 0 && index < m_comboNurbs->count())
	{


		if(m_view3D){
			m_view3D->selectNurbs(index);

		}

	}
}

void NurbsWidget::addView3d(ViewQt3D* view3d)
{
	m_listeView3D.push_back(view3d);

}

void NurbsWidget::removeView3d(ViewQt3D* view3d)
{
	m_listeView3D.removeOne(view3d);
 }

void NurbsWidget::setView3D(ViewQt3D* view3d)
{


	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}

	if(m_view3D != view3d)
	{
		m_view3D = view3d;


		connect(m_view3D,SIGNAL(sendNurbsName(QString)),m_pInstance,SLOT(receiveNameNurbs(QString)));
		connect(m_view3D->getManagerNurbs(),SIGNAL(sendColorNurbs(QColor,QColor,int,bool,int)),m_pInstance,SLOT(receiveColorNurbs(QColor,QColor,int,bool,int)));

	//	connect(m_view3D->getManagerNurbs(),SIGNAL(generateDirectrice(QString,QVector<PointCtrl>,QColor,bool)),m_sceneEditorRGB,SLOT(createListBezierPath(QString, QVector<PointCtrl>,QColor,bool)));

	/*	for(int i=0;i<m_listeView2D.count();i++)
		{
			connect(m_view3D->getManagerNurbs(),SIGNAL(generateDirectrice(QString,QVector<PointCtrl>,QColor,bool)),dynamic_cast<GraphicSceneEditor *> (m_listeView2D[i]->scene()),SLOT(createListBezierPath(QString, QVector<PointCtrl>,QColor,bool)));
		}*/

	}

}



void NurbsWidget::setView2D(Abstract2DInnerView* view2d)
{

	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}
//	qDebug()<<" VIEW 2D :"<<view2d->getBaseTitle();

	if(!m_listeView2D.contains(view2d))
	{
		connect((dynamic_cast<GraphicSceneEditor *> (view2d->scene())),SIGNAL(sendDirectriceOk()),m_pInstance,SLOT(activerDirectrice()));
		m_listeView2D.push_back(view2d);
	}

	if(view2d->getBaseTitle() =="Basemap 2")
	{
		m_sceneEditorSpectrum = (dynamic_cast<GraphicSceneEditor *> (view2d->scene()));
	}
/*	if(view2d->getBaseTitle() =="Basemap 3")
	{
		m_view2DGCC = view2d;
	}
	if(view2d->getBaseTitle() =="Basemap 5")
	{
		m_view2Dmean = view2d;
	}*/


	if(/*m_view2D != view2d &&*/ view2d->getBaseTitle() =="Basemap 6")
	{
		m_view2D = view2d;

		m_sceneEditorRGB = (dynamic_cast<GraphicSceneEditor *> (m_view2D->scene()));
		//qDebug()<<view2d->getBaseTitle()<<" , setView2D: "<<m_sceneEditor;
	//	connect((dynamic_cast<GraphicSceneEditor *> (m_view2D->scene())),SIGNAL(sendDirectriceOk()),m_pInstance,SLOT(activerDirectrice()));
		connect(m_view2D,SIGNAL(selectedNurbs(QString)),m_pInstance,SLOT(setSelectNurbs(QString)));
		connect(m_view2D,SIGNAL(deletedNurbs(QString)),m_pInstance,SLOT(setDeleteNurbs(QString)));
		//connect(m_view2D,SIGNAL(deletedDirectriceNurbs(QString)),m_pInstance,SLOT(DeleteDNurbs(QString)));


		//connect(m_view3D->getManagerNurbs(),SIGNAL(generateDirectrice(QString,QVector<PointCtrl>,QColor,bool)),m_sceneEditor,SLOT(createListBezierPath(QString, QVector<PointCtrl>,QColor,bool)));

		//qDebug()<<"chargement des nurbs";
	}





/*	connect(view2d,SIGNAL(addNurbsPoints(QVector<QPointF>,bool )),this,SLOT(receivePointsNurbs(QVector<QPointF>,bool)));
	connect(view2d,SIGNAL(updateNurbsPoints(QVector<QPointF>,bool)),this,SLOT(updatePointsNurbs(QVector<QPointF>,bool)));

	connect(view2d,SIGNAL(selectedNurbs(QString)),this,SLOT(setSelectNurbs(QString)));
	connect(view2d,SIGNAL(deletedNurbs(QString)),this,SLOT(setDeleteNurbs(QString)));

	connect(view2d,SIGNAL(signalRandomView(bool,QVector<QPointF>)),this,SLOT(showRandomView(bool, QVector<QPointF>)));
	connect(view2d,SIGNAL(signalRandomView(bool,GraphEditor_LineShape*, RandomLineView*)),this,SLOT(showRandomView(bool,GraphEditor_LineShape*, RandomLineView*)));

	connect(view2d,SIGNAL(signalRandomViewDeleted(RandomLineView*)),this,SLOT(destroyRandomView(RandomLineView*)));

*/
}

void NurbsWidget::setInlineView(Abstract2DInnerView* viewInline)
{

	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}

	RandomLineView* randomView = dynamic_cast<RandomLineView*>(viewInline);

	//if(!m_viewInline)
	if(randomView)
	{

		m_viewInline = viewInline;

		//qDebug()<<" m_viewInline->getBaseTitle() "<<m_viewInline->getBaseTitle();
		m_sceneEditor2 = (dynamic_cast<GraphicSceneEditor *> (m_viewInline->scene()));
		connect(m_sceneEditor2,SIGNAL(sendDirectriceOk()),m_pInstance,SLOT(activerGeneratrice()));
		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);


	}

}


void NurbsWidget::setModeEditable(bool edit)
{

	if(edit==true)
	{
		//validerGeneratrice->setVisible(false);
		//validerDirectrice->setVisible(false);

		//m_buttonDirectrice->setVisible(true);
		m_buttonNurbs->setVisible(true);
		m_sliderPrecision->setVisible(true);
		labelQuality->setVisible(true);

		//m_editButton->setVisible(false);
		//label2->setVisible(true);
		label3->setVisible(true);
		labelDirectrice->setVisible(true);
		labelGeneratrice->setVisible(true);
		saveNurbs->setVisible(true);
	}
	else
	{
		validerGeneratrice->setVisible(false);
		validerDirectrice->setVisible(false);

		//m_buttonDirectrice->setVisible(false);
		m_buttonNurbs->setVisible(false);
		m_sliderPrecision->setVisible(false);
		labelQuality->setVisible(false);

		//m_editButton->setVisible(true);
		//label2->setVisible(false);
		label3->setVisible(false);
		labelDirectrice->setVisible(false);
		labelGeneratrice->setVisible(false);
		saveNurbs->setVisible(false);
	}
}

void NurbsWidget::editerNurbs()
{
	m_pInstance->m_directriceOK= false;

	 m_view3D->importNurbs(/*m_view2D->getHorizonBuffer()*/getHorizonBufferValid(),m_comboNurbs->currentText()+".txt");
	 setModeEditable(true);
	 GraphicToolsWidget::showPalette("");

	// if(m_sceneEditorRGB != nullptr)
		// m_sceneEditorRGB->cloneDirectrice(m_pInstance->getDirectriceBezier());//m_sceneEditorRGB->getSelectedBezier());

	 validerGeneratrice->setVisible(true);
	 validerDirectrice->setVisible(true);
}



void NurbsWidget::generateDirectrice()
{
	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}
//	if( m_sceneEditorRGB->selectedItems().count()> 0)
	if(m_pInstance->getDirectriceBezier() != nullptr)
	{
		//if(m_pInstance->m_directriceOK==false)
		//{

			bool res = false;
			if(m_indexView != -1 && m_listeView2D[m_indexView] != nullptr)
			{
			 res = (dynamic_cast<GraphicSceneEditor *> (m_listeView2D[m_indexView]->scene()))->createOrgthognalRandomView(m_comboNurbs->currentText(),m_nurbsColor);
			}


			/*if(m_sceneEditorSpectrum!= nullptr)
			{

			}*/
			if(res== true)
			{
				m_pInstance->validerDirectrice->setIcon(QIcon(QString(":/slicer/icons/validate.png")));validerDirectrice->setIcon(QIcon(QString(":/slicer/icons/validate.png")));
				m_pInstance->m_directriceOK =true;

				if(m_view2D != nullptr) m_view2D->setNameItem(m_comboNurbs->currentText());

				m_pInstance->m_generatriceOK = false;
				m_pInstance->buttonSupprDirectrice->setVisible(true);

				if(m_sceneEditor2 != nullptr) m_sceneEditor2->setColorCustom(m_nurbsColor);
			}
	//	}
		//validerGeneratrice->setEnabled(true);
	}
	else
	{
		//qDebug()<<m_sceneEditorRGB<<" directrice not selected!!!!!"<< m_sceneEditorRGB->selectedItems().count();
	}



}

void NurbsWidget::addGeneratrice()
{
	if(m_sceneEditor2!= nullptr &&  m_sceneEditor2->selectedItems().count()> 0)
	{
		if(m_generatriceOK == false)
		{
			m_sceneEditor2->nurbs3d(m_comboNurbs->currentText());
			validerGeneratrice->setIcon(QIcon(QString(":/slicer/icons/validate.png")));
			m_generatriceOK = true;
			buttonSupprGeneratrice->setVisible(true);
		}

	}
	else
		{
			qDebug()<<" generatrice not selected!!!!!";
		}


}

int NurbsWidget::getNbNurbs()
{
	return m_listeLoadNurbs.count();
}


QString NurbsWidget::getPath(int index)
{
	return m_listeLoadNurbs[index];
}

QString NurbsWidget::getName(int index)
{
	return m_listeNameNurbs[index];
}

QString NurbsWidget::newNurbsSimple()
{
	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}
	widgetNameForSave* widget = new widgetNameForSave("Nurbs",m_pInstance);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();

		if(nom!="" )
		{

			m_pInstance->m_comboNurbs->addItem(nom);

			IsoSurfaceBuffer buffer = m_pInstance->getHorizonBufferValid();

			if(buffer.isValid())
			{
				if(m_view3D!=nullptr) m_view3D->createNurbsSimple(nom,buffer);



			}
		}
		return nom;
	}
	return "";
}


void NurbsWidget::newNurbs()
{
	widgetNameForSave* widget = new widgetNameForSave("Nurbs",this);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();

		if(nom!="" )
		{
			m_pInstance->m_directriceOK= false;
			validerDirectrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
			validerDirectrice->setVisible(false);

			validerGeneratrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
			validerGeneratrice->setVisible(false);

			/*m_directriceColor = QColor(255,255,0,255);
			QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_directriceColor.red())+","+QString::number(m_directriceColor.green())+","+QString::number(m_directriceColor.blue())+")}";
			m_buttonDirectrice->setStyleSheet(namecolor);
*/
			//QColor color3(0,0,255,255);
			QString namecolor3= "QPushButton {background-color: rgb("+QString::number(m_nurbsColor.red())+","+QString::number(m_nurbsColor.green())+","+QString::number(m_nurbsColor.blue())+")}";
			m_buttonNurbs->setStyleSheet(namecolor3);

			m_comboNurbs->addItem(nom);

			m_comboNurbs->setCurrentIndex(m_comboNurbs->count()-1);


			buttonSupprDirectrice->setVisible(false);
			buttonSupprGeneratrice->setVisible(false);

			//m_directriceOK=false;
			GraphicToolsWidget::showPalette("");

			m_sceneEditorRGB->setColorCustom(m_nurbsColor);
			setModeEditable(true);
			//m_view3D->exportNurbsObj(nom);
		}
	}
	else
	{

	}
}

void NurbsWidget::addNurbs(QString path,QString name)
{
	// m_view3D->importNurbs(m_view2D->getHorizonBuffer(),name);

	// m_comboNurbs->addItem(name);


	if (!m_pInstance)
		{
			m_pInstance = new NurbsWidget();
		}

	if(!m_listeNameNurbs.contains(name) )
	{
		//qDebug()<<" addNurbs :"<<name;
		m_pInstance->m_listeLoadNurbs.push_back(path);
		m_pInstance->m_listeNameNurbs.push_back(name);


		if(m_view3D!=nullptr)
		{

			QString nameNurbs =name.replace(".txt","");
			//

			m_view3D->importNurbsObj(path,nameNurbs);
			m_pInstance->m_comboNurbs->addItem(nameNurbs);

			m_pInstance->m_comboNurbs->setCurrentIndex(m_pInstance->m_comboNurbs->count()-1);
		}

	}



	//qDebug()<<" nb "<<m_listeNameNurbs.count();
/*	for( int i=0;i<m_listeNameNurbs.count();i++)
		{
			qDebug()<<i<< "m_listeNameNurbs"<<m_listeLoadNurbs[i];

		}
*/

}

void NurbsWidget::editerNurbs(QString path,QString name)
{


	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}

	int index =m_pInstance->m_comboNurbs->findText(name);
	if(index>=0)
	{
		m_pInstance->m_comboNurbs->setCurrentIndex(index);

		// m_pInstance->m_directriceOK= false;
		 IsoSurfaceBuffer buffer = getHorizonBufferValid();


		 if(buffer.isValid())
		 {
			// qDebug()<<"editer nurbs m_pInstance->m_indexView : "<<m_pInstance->m_indexView;


			 if(first==true)
			 {
				connect(m_view3D->getManagerNurbs(),SIGNAL(generateDirectrice(QString,QVector<PointCtrl>,QColor,bool)),dynamic_cast<GraphicSceneEditor *> (m_listeView2D[m_pInstance->m_indexView]->scene()),SLOT(createListBezierPath(QString, QVector<PointCtrl>,QColor,bool)));
				first =false;
			 }
			 m_view3D->importNurbs(buffer, m_pInstance->m_comboNurbs->currentText()+".txt");
			 m_pInstance->setModeEditable(true);
			 GraphicToolsWidget::showPalette("");

			 if(m_pInstance->getDirectriceBezier() != nullptr)
			 {
				 	 //getCurrentBezier
				 GraphEditor_Path* path  =m_pInstance->getDirectriceBezier(); //m_sceneEditorRGB->getSelectedBezier();

				 //if(path) qDebug()<<name <<"path name "<<path->getNameNurbs;
				// connect(m_view3D->getManagerNurbs(),SIGNAL(generateDirectrice(QString,QVector<PointCtrl>,QColor,bool)),dynamic_cast<GraphicSceneEditor *> (m_listeView2D[m_pInstance->m_indexView]->scene()),SLOT(createListBezierPath(QString, QVector<PointCtrl>,QColor,bool)));

				 if(path && path->getNameNurbs() == name) //path != nullptr)
				{

					 //m_sceneEditorRGB->cloneDirectrice(m_sceneEditorRGB->getSelectedBezier());
					 m_pInstance->validerGeneratrice->setVisible(true);
					 m_pInstance-> validerDirectrice->setVisible(true);
				}
				else
				{
					m_nurbsColor = QColor(180,180,180,255);
					QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_nurbsColor.red())+","+QString::number(m_nurbsColor.green())+","+QString::number(m_nurbsColor.blue())+")}";
						m_pInstance->m_buttonNurbs->setStyleSheet(namecolor);
					m_sceneEditorRGB->setColorCustom(m_nurbsColor);
					 m_pInstance->validerGeneratrice->setVisible(false);
					 m_pInstance->validerDirectrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
					 m_pInstance->validerGeneratrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
					m_pInstance-> validerDirectrice->setVisible(false);
					m_pInstance->buttonSupprGeneratrice->setVisible(false);
					m_pInstance-> buttonSupprDirectrice->setVisible(false);

				}
			 }

			// m_pInstance->validerGeneratrice->setVisible(true);
			// m_pInstance-> validerDirectrice->setVisible(true);
		 }

	}

}

IsoSurfaceBuffer NurbsWidget::getHorizonBufferValid()
{
	if (!m_pInstance)
		{
			m_pInstance = new NurbsWidget();
		}
	for(int i=0;i<m_listeView2D.count();i++)
	{
		if(m_listeView2D[i] != nullptr)
		{
			IsoSurfaceBuffer buffer = m_listeView2D[i]->getHorizonBuffer();
			if(buffer.isValid())
			{
				//qDebug()<<"getHorizonBufferValid : "<<i;
				m_pInstance->m_indexView=i;
				return buffer;
			}
		}
	}

	IsoSurfaceBuffer buffer2;
	return buffer2;
}



GraphEditor_Path* NurbsWidget::getDirectriceBezier()
{
	for(int i=0;i<m_listeView2D.count();i++)
	{
		if(m_listeView2D[i] != nullptr)
		{
			GraphEditor_Path* path  =(dynamic_cast<GraphicSceneEditor *> (m_listeView2D[i]->scene()))->getSelectedBezier();
			if(path != nullptr)
			{
				m_indexView=i;
				//qDebug()<<"getHorizonBufferValid : "<<i;
				return path;
			}
		}
	}


	return nullptr;
}

void NurbsWidget::removeNurbs(QString path,QString name)
{

	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}

	 if(m_sceneEditorRGB != nullptr)
		 m_sceneEditorRGB->clearDirectrice(name);

	/*int index = m_pInstance->m_comboNurbs->findText(name);
		if(index != -1) m_pInstance->m_comboNurbs->removeItem(index);

	m_view3D->destroyNurbs(name);*/


	 if(m_pInstance->m_indexView != -1 && m_listeView2D[m_pInstance->m_indexView] != nullptr)
	 {
	 	 (dynamic_cast<GraphicSceneEditor *> (m_listeView2D[m_pInstance->m_indexView]->scene()))->clearDirectrice(name);
	 }

	m_sceneEditor2 = nullptr;


	//TODO
	m_view3D->destroyNurbs(name);
	int index = m_pInstance->m_comboNurbs->findText(name);
	if(index != -1) m_pInstance->m_comboNurbs->removeItem(index);

	int indice = m_pInstance->m_listeNameNurbs.indexOf(name+".txt");

	if(indice >=0)
	{
		m_pInstance->m_listeLoadNurbs.removeAt(indice);
		m_pInstance->m_listeNameNurbs.removeAt(indice);
	}

}


/*
void NurbsWidget::importNurbs()
{

	qDebug()<<" OBSOLETE";
	 m_view3D->importNurbs(m_view2D->getHorizonBuffer(),"nurbs12300.txt");

	 m_comboNurbs->addItem("test");
}*/

void NurbsWidget::addCombo(QString name)
{
	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}
	m_pInstance->m_comboNurbs->addItem(name);
}

void NurbsWidget::clearCombo()
{
	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}
	m_pInstance->m_comboNurbs->clear();
}


void NurbsWidget::exportNurbs()
{
	/*widgetNameForSave* widget = new widgetNameForSave(this);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();

		if(nom!="" )
		{
			m_view3D->exportNurbsObj(nom);
		}
	}*/

	if( m_comboNurbs->currentText()!= "") m_view3D->exportNurbsObj(m_comboNurbs->currentText());
}

/*
void NurbsWidget::setDirectriceColor()
{
	 QColor color = QColorDialog::getColor(m_directriceColor, this );
	if( color.isValid() )
	{
		m_directriceColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_directriceColor.red())+","+QString::number(m_directriceColor.green())+","+QString::number(m_directriceColor.blue())+")}";
		m_buttonDirectrice->setStyleSheet(namecolor);

		m_view3D->setColorDirectrice(m_directriceColor);
		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
		if(randomView)randomView->setColorCross(m_directriceColor);
		else qDebug()<<" No find RandomLineView";

	}
}*/

void NurbsWidget::setNurbsColor()
{
	 QColor color = QColorDialog::getColor(m_nurbsColor, this );
	if( color.isValid() )
	{
		m_nurbsColor = color;
		QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_nurbsColor.red())+","+QString::number(m_nurbsColor.green())+","+QString::number(m_nurbsColor.blue())+")}";
		m_buttonNurbs->setStyleSheet(namecolor);

		m_view3D->setColorNurbs(color);

		if(m_sceneEditor2!= nullptr)
			m_sceneEditor2->setGeneratriceColor(m_nurbsColor,m_comboNurbs->currentText());
	}
}

void NurbsWidget::setColor(QString s, QColor col)
{
	if (!m_pInstance)
	{
		m_pInstance = new NurbsWidget();
	}
	m_pInstance->setSelectNurbs(s);

	m_nurbsColor = col;
	QString namecolor= "QPushButton {background-color: rgb("+QString::number(m_nurbsColor.red())+","+QString::number(m_nurbsColor.green())+","+QString::number(m_nurbsColor.blue())+")}";
	m_pInstance->m_buttonNurbs->setStyleSheet(namecolor);

	m_view3D->setColorNurbs(s,m_nurbsColor);

	if(m_sceneEditor2!= nullptr)
		m_sceneEditor2->setGeneratriceColor(m_nurbsColor,m_pInstance->m_comboNurbs->currentText());

}

void NurbsWidget::saveColor(QString path, QColor col)
{
	bool valid;
	Manager::NurbsParams params = Manager::read(path, &valid);
	if (valid)
	{
		params.color = col;
		valid = Manager::write(path, params);
	}
}

QColor NurbsWidget::getCurrentColor()
{
	return m_nurbsColor;
}

void NurbsWidget::setNurbsPrecision(int val)
{
	m_view3D->setPrecisionNurbs(val);
}

void NurbsWidget::nurbsSelectedChanged(int index)
{
	//m_view3D->selectNurbs(index);//todo a remettre
}

void NurbsWidget::receiveNameNurbs(QString name)
{
	//m_comboNurbs->addItem(name);
	//qDebug()<<"**** receiveNameNurbs "<<name;
	if(m_indexView != -1 && m_listeView2D[m_indexView] != nullptr)
	{
		m_listeView2D[m_indexView]->setNameItem(name);
	}
}

void NurbsWidget::receiveColorNurbs(QColor colorDir,QColor colorNurbs,int precision,bool editable,int layer)
{
	//QString namecolor2= "QPushButton {background-color: rgb("+QString::number(colorDir.red())+","+QString::number(colorDir.green())+","+QString::number(colorDir.blue())+")}";
	//m_buttonDirectrice->setStyleSheet(namecolor2);

	QString namecolor3= "QPushButton {background-color: rgb("+QString::number(colorNurbs.red())+","+QString::number(colorNurbs.green())+","+QString::number(colorNurbs.blue())+")}";
	m_buttonNurbs->setStyleSheet(namecolor3);


	if(precision>0) m_sliderPrecision->setValue(precision);


	setModeEditable(editable);


	m_nurbsColor = colorNurbs;


	//m_labelLayer->setText("Layer: "+QString::number(layer));

}


void NurbsWidget::setSelectNurbs(QString s)
{
	int index = m_comboNurbs->findText(s);
	if(index >=0 )m_comboNurbs->setCurrentIndex(index);
	m_view3D->selectNurbs(s);
}

void NurbsWidget::setDeleteNurbs(QString s)
{
	m_view3D->deleteNurbs(s);
	int index = m_comboNurbs->findText(s);
	if(index != -1) m_comboNurbs->removeItem(index);
}


void NurbsWidget::receiveCrossPoints(QVector<PointCtrl> listeCtrls,QVector<QPointF>  listepoints,bool isopen,QPointF cross)
{

	SingleSectionView* sectionView = dynamic_cast<SingleSectionView*>(m_viewInline);
	QVector<QVector3D>  listePts3D;
	if(sectionView != nullptr)
	{

		for(int i=0;i<listepoints.count();i++)
		{
			QVector3D posTr = sectionView->viewWorldTo3dWord(listepoints[i]);
			QVector3D swap(posTr.x(),posTr.z(),posTr.y());
			QVector3D pos = m_view3D->sceneTransform() * swap;
			listePts3D.append(pos);

		}
	}
	else
	{
		RandomLineView* randomView = dynamic_cast<RandomLineView*>(m_viewInline);
		if(randomView != nullptr)
		{

			for(int i=0;i<listepoints.count();i++)
			{
				QVector3D posTr = randomView->viewWorldTo3dWordExtended(listepoints[i]);
				QVector3D swap(posTr.x(),posTr.z(),posTr.y());
				QVector3D pos = m_view3D->sceneTransform() * swap;
				listePts3D.append(pos);
			}
		}
	}

	// for(int i=0;i<listeCtrls.count();i++)
	//			qDebug()<<i<<" ,==> pos : "<<listeCtrls[i]->m_position;

	qDebug()<<" OBSOLETE";
	//m_view3D->createSection(listeCtrls, listePts3D,-1,isopen,true,cross);//m_indexCurrentPts ==  -1
}



void NurbsWidget::activerDirectrice()
{

	m_pInstance->validerDirectrice->setVisible(true);
}

void NurbsWidget::activerGeneratrice()
{
	m_pInstance->validerGeneratrice->setVisible(true);
}

void NurbsWidget::supprimerDirectrice()
{
	QMessageBox::StandardButton res = QMessageBox::question(this,"Confirmation delete","Are you sure to delete the directrice?");
	if (res ==QMessageBox::Yes)
	{
		validerDirectrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
		buttonSupprDirectrice->setVisible(false);
		buttonSupprGeneratrice->setVisible(false);
		validerGeneratrice->setVisible(false);

		m_sceneEditorRGB->directriceDeleted(m_comboNurbs->currentText());

		m_pInstance->m_directriceOK =false;
		m_generatriceOK= false;

		m_view3D->destroyNurbs(m_comboNurbs->currentText());
	}
}


void NurbsWidget::supprimerGeneratrice()
{
	QMessageBox::StandardButton res = QMessageBox::question(this,"Confirmation delete","Are you sure to delete the generatrice?");
		if (res ==QMessageBox::Yes)
		{
			validerGeneratrice->setIcon(QIcon(QString(":/slicer/icons/validateBlanc.png")));
			buttonSupprGeneratrice->setVisible(false);
			if(m_sceneEditor2 != nullptr) m_sceneEditor2->supprimerNurbs3d(m_comboNurbs->currentText());
			m_generatriceOK= false;

		}
}


