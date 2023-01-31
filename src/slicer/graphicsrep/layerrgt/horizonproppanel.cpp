#include "horizonproppanel.h"

#include <iostream>
//#include "horizonfolderdatarep.h"
#include "horizondatarep.h"
#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cudargbinterleavedimage.h"

#include "LayerSlice.h"

#include <QGroupBox>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QAction>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QComboBox>
#include <QListWidget>
#include <QMessageBox>
#include "abstractinnerview.h"
#include "pointpickingtask.h"
#include "idata.h"
#include "horizonfolderdata.h"
//#include "fixedrgblayersfromdatasetandcube.h"
#include "orderstackhorizonwidget.h"
#include "rgbpalettewidget.h"
#include "freehorizon.h"

#include <algorithm>
#include <functional>

HorizonPropPanel::HorizonPropPanel(HorizonDataRep *rep,  QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;

	QVBoxLayout * mainLayout01 = new QVBoxLayout(this);

		QHBoxLayout * mainLayout02 = new QHBoxLayout;


		m_orderListWidget = new QListWidget;
		m_orderListWidget->setSelectionMode(QAbstractItemView::SingleSelection);

		m_animTimer =new QTimer();
		connect(m_animTimer,SIGNAL(timeout()),this,SLOT(updateAnimation()));

		int nbelem = m_rep->horizonFolderData()->completOrderList().count();



		for(int i=0;i<nbelem;i++)
		{
			FreeHorizon* fixed = dynamic_cast<FreeHorizon*>(m_rep->horizonFolderData()->completOrderList()[i]);
			if(fixed != nullptr)
			{
				m_orderListWidget->addItem(new QListWidgetItem(fixed->name()));
			}
			else
				qDebug()<<"  FreeHorizon est null";

		}

		mainLayout02->addWidget(m_orderListWidget);


		QVBoxLayout * mainLayout04 = new QVBoxLayout;
		QToolButton *pushbutton_up = new QToolButton();
		QPixmap pixmap0(style()->standardPixmap(QStyle::SP_MediaPlay ));
		QTransform tr0;
		tr0.rotate(-90);
		pixmap0 = pixmap0.transformed(tr0);
		pushbutton_up->setIcon(pixmap0);


		QToolButton *pushbutton_down = new QToolButton();
		QPixmap pixmap1(style()->standardPixmap(QStyle::SP_MediaPlay ));
		QTransform tr1;
		tr1.rotate(90);
		pixmap1 = pixmap1.transformed(tr1);
		pushbutton_down->setIcon(pixmap1);


		QToolButton *pushbutton_top = new QToolButton();
		QPixmap pixmap(style()->standardPixmap(QStyle::SP_MediaSkipBackward ));
		QTransform tr;
		tr.rotate(90);
		pixmap = pixmap.transformed(tr);
		pushbutton_top->setIcon(pixmap);

		QToolButton *pushbutton_bottom = new QToolButton();
		QPixmap pixmap2(style()->standardPixmap(QStyle::SP_MediaSkipForward ));
		QTransform tr2;
		tr2.rotate(90);
		pixmap2 = pixmap2.transformed(tr2);
		pushbutton_bottom->setIcon(pixmap2);

		mainLayout04->addWidget(pushbutton_top);
		mainLayout04->addWidget(pushbutton_up);
		mainLayout04->addWidget(pushbutton_down);
		mainLayout04->addWidget(pushbutton_bottom);
		mainLayout02->addLayout(mainLayout04);

		QHBoxLayout* lay0 = new QHBoxLayout();
		QLabel* attributelabel = new QLabel("Attributs");
		//m_comboView3D = new QComboBox();
		m_comboAttribut = new QComboBox();
		fillComboAttribut();
		lay0->addWidget(attributelabel);
	//	lay0->addWidget(m_comboView3D);
		lay0->addWidget(m_comboAttribut);





		QHBoxLayout* lay1 = new QHBoxLayout();
		m_animSlider = new QSlider(Qt::Orientation::Horizontal);
		m_animSlider->setMinimum(0);
		m_animSlider->setMaximum(nbelem-1);

		m_animSlider->setTickInterval(1);
		m_animSlider->setSingleStep(1);
		m_animSlider->setTracking(false);


		m_playButton = new QToolButton();
		m_playButton->setCheckable(true);
		m_playButton->setIcon(style()->standardPixmap(QStyle::SP_MediaPlay ));

		m_saveButton = new QToolButton();
		m_saveButton->setIcon(style()->standardPixmap(QStyle::SP_DialogSaveButton ));

		lay1->addWidget(m_animSlider);
		lay1->addWidget(m_playButton);
		//lay1->addWidget(m_saveButton);


		//QPushButton* cacheButton = new QPushButton("Cache");
		//connect(cacheButton,SIGNAL(clicked()),this,SLOT(computeCache()));

		QHBoxLayout* lay2 = new QHBoxLayout();
		QLabel* speedlabel = new QLabel("Speed");
		m_speedSlider = new QSlider(Qt::Orientation::Horizontal);
		m_speedSlider->setMinimum(100);
		m_speedSlider->setMaximum(1000);
		m_speedSlider->setValue(m_speedAnim);
		m_speedSlider->setTickInterval(100);
		m_speedSlider->setSingleStep(1);
		m_speedSlider->setTracking(false);

		lay2->addWidget(speedlabel);
		lay2->addWidget(m_speedSlider);

	//	lay2->addWidget(cacheButton);



		mainLayout01->addLayout(mainLayout02);
		mainLayout01->addLayout(lay2);
		mainLayout01->addLayout(lay0);
		mainLayout01->addLayout(lay1);


		//Palettes
		m_paletteRGB = new RGBPaletteWidget(this);
		m_paletteRGB->hide();
		mainLayout01->addWidget(m_paletteRGB);


		m_palette = new PaletteWidget(this);
		m_palette->hide();

		mainLayout01->addWidget(m_palette);


		m_lockPalette = new QCheckBox("Lock Palette");
		m_lockPalette->hide();
			updateLockCheckBox();
			mainLayout01->addWidget(m_lockPalette, 0, Qt::AlignmentFlag::AlignTop);

			connect(m_lockPalette, &QCheckBox::stateChanged, this,
					&HorizonPropPanel::lockPalette);
			connect(m_palette, &PaletteWidget::rangeChanged,
						this, &HorizonPropPanel::updateLockRange);

/*			connect(m_rep->image(),
					QOverload<const QVector2D&>::of(&CUDAImagePaletteHolder::rangeChanged), this,
					&HorizonPropPanel::updateLockRange);
*/


		//mainLayout01->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);

	/*	m_palette->setPaletteHolders(m_rep->horizonFolderData()->image()->holders());
		m_palette->setOpacity(m_rep->horizonFolderData()->image()->opacity());




		//Connect the image update
		connect(m_palette, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),
				m_rep->horizonFolderData()->image(),
				SLOT(setRange(unsigned int ,const QVector2D & )));
		connect(m_palette, SIGNAL(opacityChanged(float)),
				m_rep->horizonFolderData()->image(), SLOT(setOpacity(float)));

		connect(m_rep->horizonFolderData()->image(),
				SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_palette,
				SLOT(setRange(unsigned int ,const QVector2D & )));
		connect(m_rep->horizonFolderData()->image(), SIGNAL(opacityChanged(float)),
				m_palette, SLOT(setOpacity(float)));

*/

		connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));


		connect(pushbutton_up, SIGNAL(clicked()), this, SLOT(moveUp()));
		connect(pushbutton_down, SIGNAL(clicked()), this, SLOT(moveDown()));
		connect(pushbutton_top, SIGNAL(clicked()), this, SLOT(moveTop()));
		connect(pushbutton_bottom, SIGNAL(clicked()), this, SLOT(moveBottom()));


		connect(m_animSlider, SIGNAL(valueChanged(int)),this,SLOT(moveAnimation(int)));
		connect(m_animSlider, SIGNAL(sliderMoved(int)),this,SLOT(moveAnimation(int)));
		connect(m_playButton, SIGNAL(toggled(bool)),this,SLOT(playAnimation(bool)));
		connect(m_saveButton, SIGNAL(toggled(bool)),this,SLOT(saveAnimation(bool)));

		connect(m_speedSlider, SIGNAL(valueChanged(int)),this,SLOT(speedChanged(int)));

		//connect(m_comboView3D,SIGNAL(currentIndexChanged(int)),this,SLOT(view3DChanged(int)));
		connect(m_comboAttribut,SIGNAL(currentIndexChanged(int)),this,SLOT(attributChanged(int)));



		connect(m_rep->data(),SIGNAL(layerAdded(IData*)),this,SLOT(dataAdded(IData*)));
		connect(m_rep->data(),SIGNAL(layerRemoved(IData*)),this,SLOT(dataRemoved(IData*)));
		connect(m_rep->data(),SIGNAL(orderChanged(int, int)), this, SLOT(orderChangedFromData(int ,int)));

	  /*  int index = 0;
		    int index3D = 1;
		    while(index<m_graphicsview->getInnerViews().size()) {//}&& dynamic_cast<ViewQt3D*>(m_graphicsview->getInnerViews()[index])==nullptr) {

		    	ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(m_graphicsview->getInnerViews()[index]);
		    	if(view3D !=nullptr)
		    	{
		    		//m_comboView3D->addItem("View3D "+QString::number(m_index3D));
		    		//m_viewIndex.push_back(index);
		    		//m_viewAttribIndex.push_back(qMakePair(m_index3D,0));
		    		m_index3D++;
		    		  qDebug()<<"-------------> view index :"<<view3D->getName();
		    	}
		    	index++;
		    }*/

/*		int indexDefault = m_comboAttribut->findText("spectrum",Qt::MatchContains);
		if(indexDefault>=0 )
		{
			m_comboAttribut->setCurrentIndex(indexDefault);

			m_animSlider->setValue(0);
		}*/

}
HorizonPropPanel::~HorizonPropPanel() {

}
/*

void HorizonPropPanel::computeCache()
{
	m_cacheGPU=true;
	m_rep->computeCache();

}*/

void HorizonPropPanel::setRangeLock(unsigned int i,const QVector2D &range)
{
	if(m_lockPalette->isChecked())
	{
		if(i==0) m_rep->horizonFolderData()->lockRange(range, m_rep->horizonFolderData()->lockedRangeGreen(m_rep->getNameAttribut()), m_rep->horizonFolderData()->lockedRangeBlue(m_rep->getNameAttribut()) ,m_rep->getNameAttribut(),m_modeRGB);
		if(i==1) m_rep->horizonFolderData()->lockRange(m_rep->horizonFolderData()->lockedRangeRed(m_rep->getNameAttribut()),range, m_rep->horizonFolderData()->lockedRangeBlue(m_rep->getNameAttribut()), m_rep->getNameAttribut(),m_modeRGB);
		if(i==2) m_rep->horizonFolderData()->lockRange(m_rep->horizonFolderData()->lockedRangeRed(m_rep->getNameAttribut()), m_rep->horizonFolderData()->lockedRangeGreen(m_rep->getNameAttribut()),range , m_rep->getNameAttribut(),m_modeRGB);

	}

}

void HorizonPropPanel::updateLockCheckBox() {
	bool isRangeLocked = m_rep->horizonFolderData()->isRangeLocked(m_rep->getNameAttribut());
	int lockState = (isRangeLocked) ? Qt::Checked : Qt::Unchecked;

	QSignalBlocker b1(m_lockPalette);
	m_lockPalette->setChecked(lockState);
}

void HorizonPropPanel::lockPalette(int state) {

	if(m_rep->image() == nullptr) return;
	if (state==Qt::Checked) {
		m_rep->horizonFolderData()->lockRange(m_rep->image()->redRange(),m_rep->image()->greenRange(),m_rep->image()->blueRange(),m_rep->getNameAttribut(),m_modeRGB);
	} else {
		m_rep->horizonFolderData()->unlockRange(m_rep->getNameAttribut());
	}
}

void HorizonPropPanel::updateLockRange(const QVector2D & range) {

	if (m_rep->horizonFolderData()->isRangeLocked(m_rep->getNameAttribut())) {
		m_rep->horizonFolderData()->lockRange(m_rep->image()->redRange(),m_rep->image()->greenRange(),m_rep->image()->blueRange(),m_rep->getNameAttribut(),m_modeRGB);
		//m_rep->data()->lockRange(range);
	}
}


void HorizonPropPanel::fillComboAttribut()
{
	m_comboAttribut->clear();

	QList<FreeHorizon*> listFree =m_rep->horizonFolderData()->completOrderList();

	std::function<bool(const QString&, const QString&)> comp = [](const QString& first, const QString& second) {
		return QString::localeAwareCompare(first, second)<0;
	};

	// get union and intersection
	QStringList intersectionAttributs, unionAttributs;
	for(int i=0;i<listFree.count();i++)
	{
		QStringList horizonAttributs;
		for(int j=0;j< listFree[i]->m_attribut.size();j++)
		{
			QString name = listFree[i]->m_attribut[j].name();
			if (name!="" && !horizonAttributs.contains(name))
			{
				horizonAttributs.append(name);
			}
		}

		// because set_intersection and set_union work on sorted ranges
		std::sort(horizonAttributs.begin(), horizonAttributs.end(), comp);

		if (i>0)
		{
			QStringList newIntersection;
			std::set_intersection(horizonAttributs.begin(), horizonAttributs.end(),
					intersectionAttributs.begin(), intersectionAttributs.end(),
					std::back_inserter(newIntersection), comp);
			intersectionAttributs = newIntersection;
		}
		else
		{
			intersectionAttributs = horizonAttributs;
		}

		QStringList newUnion;
		std::set_union(horizonAttributs.begin(), horizonAttributs.end(),
				unionAttributs.begin(), unionAttributs.end(),
				std::back_inserter(newUnion), comp);
		unionAttributs = newUnion;
	}

	// fill combobox
	for(int i=0; i<unionAttributs.count(); i++)
	{
		QString name = unionAttributs[i];
		bool inIntersection = intersectionAttributs.contains(name);
		m_comboAttribut->addItem(name, QVariant(inIntersection));
		if (!inIntersection)
		{
			m_comboAttribut->setItemData(i, QColor(Qt::red), Qt::BackgroundRole);
		}
	}
}

void HorizonPropPanel::attributChanged(int i)
{
	if(i>=0 && i<m_comboAttribut->count())
	{
		QVariant var = m_comboAttribut->currentData();
		bool allowedAttribute = var.canConvert<bool>() && var.toBool();

		if (allowedAttribute)
		{
			if(m_comboAttribut->currentText().contains("spectrum") )
				m_modeRGB = true;
			else
				m_modeRGB= false;

			m_rep->setNameAttribut(m_comboAttribut->currentText());


			m_lockPalette->show();

			if(m_modeRGB)
			{
				if( m_lastRGB == false)
				{
					disconnect(m_palette, SIGNAL(rangeChanged(const QVector2D & )),this,SLOT(setRangeToImage(const QVector2D & )));

					if(m_lastimage)
					{
						disconnect(m_palette, SIGNAL(opacityChanged(float)),m_lastimage, SLOT(setOpacity(float)));
						disconnect(m_lastimage, SIGNAL(opacityChanged(float)),m_palette, SLOT(setOpacity(float)));
					}
				}

				m_paletteRGB->show();
				m_palette->hide();
			}
			else
			{
				if( m_lastRGB == true)
				{
					connect(m_palette, SIGNAL(rangeChanged(const QVector2D & )),this,SLOT(setRangeToImage(const QVector2D & )));

					if(m_lastimage)
					{
						disconnect(m_paletteRGB, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),m_lastimage,SLOT(setRange(unsigned int ,const QVector2D & )));
						disconnect(m_paletteRGB, SIGNAL(opacityChanged(float)),m_lastimage, SLOT(setOpacity(float)));

						disconnect(m_lastimage,SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_paletteRGB,SLOT(setRange(unsigned int ,const QVector2D & )));
						disconnect(m_lastimage, SIGNAL(opacityChanged(float)),m_paletteRGB, SLOT(setOpacity(float)));
					}
				}
				m_paletteRGB->hide();
				m_palette->show();
			}

			dynamic_cast<HorizonFolderData*>(m_rep->data())->setCurrentData(m_animSlider->value());

			m_lastRGB = m_modeRGB;
		}
		else
		{
			QString attributName = m_comboAttribut->currentText();

			m_rep->setNameAttribut("");

			// without that the palettes are not hidden
			m_paletteRGB->hide();
			m_palette->hide();

			warnBadAttribut(attributName);
		}
	}
	else
	{
		m_rep->setNameAttribut("");
	}

}

void HorizonPropPanel::add()
{

}


void HorizonPropPanel::remove()
{

}

void HorizonPropPanel::saveAnimation(bool actif)
{
	widgetNameForSave* widget = new widgetNameForSave("Animation horizons",this);
	if ( widget->exec() == QDialog::Accepted)
	{
		QString nom = widget->getName();

		qDebug()<<" save animation Horizon";

		QString pathfile = m_rep->horizonFolderData()->getPathSave(0);
		QDir dir(pathfile);
		dir.cdUp();


		qDebug()<<" save animation Horizon"<< dir.absolutePath();

		QString directory="/Animations/";

		bool res = dir.mkpath("Animations");

		QString newfile = dir.absolutePath()+directory+nom+".hor";

		qDebug()<<"  saveAnimation : "<<newfile;

		writeAnimation(newfile);
	}



}

void HorizonPropPanel::writeAnimation(QString newfile)
{
	QFile file(newfile);
	if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
	{
		qDebug()<<" ouverture du fichier impossible "<<newfile;
		return;
	}
	QTextStream out(&file);

	int nbHorizon = m_rep->horizonFolderData()->getNbFreeHorizon();
	if(m_rep->getNameAttribut()=="")
	{
		qDebug()<<"Error attribut not selected";
		return;
	}
	qDebug()<<"m_rep->getNameAttribut()==>"<<m_rep->getNameAttribut();
	out<<"nbHorizons"<<"|"<<nbHorizon<<"\n";
	out<<"lockPalette"<<"|"<<m_lockPalette->isChecked()<<"\n";
	out<<"attributs"<<"|"<<m_rep->getNameAttribut()<<"\n";

	out<<"typePaletteRGB"<<"|"<<m_modeRGB<<"\n";
	if(m_modeRGB)
	{
		QVector2D range1 = m_paletteRGB->getRange(0);//  m_rep->horizonFolderData()->lockedRangeRed(m_rep->getNameAttribut());
		out<<"rangeRed"<<"|"<<range1.x()<<"|"<<range1.y()<<"\n";
		QVector2D range2 = m_paletteRGB->getRange(1);//m_rep->horizonFolderData()->lockedRangeGreen(m_rep->getNameAttribut());
		out<<"rangeGreen"<<"|"<<range2.x()<<"|"<<range2.y()<<"\n";
		QVector2D range3 = m_paletteRGB->getRange(2);//m_rep->horizonFolderData()->lockedRangeBlue(m_rep->getNameAttribut());
		out<<"rangeBlue"<<"|"<<range3.x()<<"|"<<range3.y()<<"\n";
	}
	else
	{
		QVector2D range1 = m_palette->getRange();
		out<<"range"<<"|"<<range1.x()<<"|"<<range1.y()<<"\n";
	}
	for(int i=0;i<nbHorizon;i++)
	{
		out<<"path"<<"|"<<m_rep->horizonFolderData()->getOrderList(i)<<"|"<<m_rep->horizonFolderData()->getPathSave(i)<<"\n";
	}

	out<<"\n";

	file.close();

}

void HorizonPropPanel::readAnimation(QString path)
{
	QFile file(path);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		qDebug()<<"Read animation ouverture du fichier impossible :"<<path;
		return;
	}


	QTextStream in(&file);

	int nbHorizon =0;
	QString attribut;
	bool paletteRGB = true;
	bool lock = true;
	QVector2D rangered,rangegreen,rangeblue,range;
	QList<QString > listepath;
	QList<int > listeindex;
	while(!in.atEnd())
	{
		QString line = in.readLine();
		QStringList linesplit = line.split("|");
		if(linesplit.count()> 0)
		{
			if(linesplit[0] =="nbHorizons")
			{
				nbHorizon= linesplit[1].toInt();

			}
			else if(linesplit[0] =="attributs")
			{
				attribut = linesplit[1];
			}
			else if(linesplit[0] =="typePaletteRGB")
			{
				int RGB = linesplit[1].toInt();
				if(RGB==1 ) paletteRGB = false;
			}
			else if(linesplit[0] =="lockPalette")
			{
				int lockP= linesplit[1].toInt();
				if(lockP==1 ) lock = false;
			}
			else if(linesplit[0] =="rangeRed")
			{
				rangered = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="rangeGreen")
			{
				rangegreen = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="rangeBlue")
			{
				rangeblue = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="range")
			{
				range = QVector2D(linesplit[1].toFloat(),linesplit[2].toFloat());
			}
			else if(linesplit[0] =="path")
			{
				int index = linesplit[1].toInt();
				QString path = linesplit[2];
				listepath.push_back(path);
				listeindex.push_back(index);
			}
		}
	}
	file.close();

	qDebug()<<" import animation ok";
}


void HorizonPropPanel::playAnimation(bool actif)
{
	if(actif)
	{
		m_cacheGPU=true;
		m_rep->horizonFolderData()->computeCache();
		m_cacheIndex=m_animSlider->maximum();
		m_animTimer->start(m_speedAnim);
	}
	else
	{
		m_animTimer->stop();
		m_lastimage = nullptr;
		m_cacheGPU=false;
		m_rep->horizonFolderData()->clearCache();
	}
}
void HorizonPropPanel::updateAnimation()
{
	if(m_cacheGPU==false)
	{
		int index = m_animSlider->value();
		//int newIndex = index+1;
	//	if(newIndex > m_animSlider->maximum()) newIndex = 0;

		int newIndex = index-1;
		if(newIndex <0) newIndex = m_animSlider->maximum();
		m_animSlider->setValue(newIndex);
	}
	else
	{
		m_cacheIndex--;
		if(m_cacheIndex <0) m_cacheIndex = m_animSlider->maximum();
		m_rep->horizonFolderData()->showCache(m_cacheIndex);

		m_animSlider->setValue(m_cacheIndex);
	}
}

void HorizonPropPanel::speedChanged(int value)
{
	m_speedAnim =m_speedSlider->maximum()+ m_speedSlider->minimum() - value;
	if(m_animTimer->isActive())
	{
		m_animTimer->setInterval(m_speedAnim);
	}
}

void HorizonPropPanel::trt_basketListSelectionChanged()
{
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();
	if (selected.size()==0)
	{
		return;
	}
	int index= m_orderListWidget->row(selected[0]);

	if(index>=0 && index !=  m_animSlider->value())
	{
	//	QSignalBlocker b1(m_animSlider);
		m_animSlider->setValue(index);
		if(m_cacheGPU==false) moveAnimation(index);
		//m_lastSelected = index;
	}
}

void HorizonPropPanel::moveAnimation(int value)
{

	if(value != m_lastValue)
	{

		if( value >=0 && value< m_orderListWidget->count() )
		{
			dynamic_cast<HorizonFolderData*>(m_rep->data())->setCurrentData(value);

			QSignalBlocker b1(m_orderListWidget);
			m_orderListWidget->setCurrentRow(value,QItemSelectionModel::ClearAndSelect);
			/*int index = m_indexList[value];
			for(int i=0;i<m_viewIndex.count();i++)
			{

				qDebug()<<" _dataOrderList[index]"<<m_dataOrderList[index]<<" , m_viewIndex[i] "<<m_viewIndex[i] ;
				m_graphicsview->SetDataItem(m_dataOrderList[index] ,m_viewIndex[i],Qt::Checked);

				m_orderListWidget->setCurrentRow(value,QItemSelectionModel::Select);

				if(m_lastSelected>=0 && index != m_indexList[m_lastSelected]  )
				{
					m_orderListWidget->setCurrentRow(m_lastSelected,QItemSelectionModel::Deselect);
					m_graphicsview->SetDataItem( m_dataOrderList[ m_indexList[m_lastSelected]],m_viewIndex[i],Qt::Unchecked);
				}
			}*/

			if(m_lastSelected>=0 )m_orderListWidget->setCurrentRow(m_lastSelected,QItemSelectionModel::Deselect);


			m_lastSelected = value;
		}
		m_lastValue = value;
	}
}

void HorizonPropPanel::moveUp()
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();
	disconnect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	// reorder
	std::sort(selected.begin(), selected.end(), [this](const QListWidgetItem* first, const QListWidgetItem* second) {
		int rowFirst = this->m_orderListWidget->row(first);
		int rowSecond = this->m_orderListWidget->row(second);
		return rowFirst<rowSecond;
	});

	// create the position post operation, since limit is -1, start from the lowest row
	int lastOutRow = -1;
	std::vector<int> outRows;
	outRows.resize(selected.count());
	for(int i=0; i<selected.count();i++)
	{
		int oldRow = m_orderListWidget->row(selected[i]);
		int newRow = oldRow;
		if (oldRow-1>lastOutRow)
		{
			newRow = oldRow-1;
		}
		lastOutRow = newRow;
		outRows[i] = newRow;
	}

	// also start from the lowest row
	for(int i=0;i<selected.count();i++)
	{
		int index= m_orderListWidget->row(selected[i]);

		if(outRows[i]!=index)
		{
			dynamic_cast<HorizonFolderData*>(m_rep->data())->changeOrder(index,outRows[i]);

			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(outRows[i],item);
		}
	}

	// reset selection
	m_orderListWidget->clearSelection();
	for (int i=0;i<selected.count();i++)
	{
		selected[i]->setSelected(true);
	}
	connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	m_orderMutex.unlock();
}

void HorizonPropPanel::moveDown()
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();
	disconnect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	// reorder
	std::sort(selected.begin(), selected.end(), [this](const QListWidgetItem* first, const QListWidgetItem* second) {
		int rowFirst = this->m_orderListWidget->row(first);
		int rowSecond = this->m_orderListWidget->row(second);
		return rowFirst<rowSecond;
	});

	// create the position post operation, since limit is m_orderListWidget->count, start from the highest row
	int lastOutRow = m_orderListWidget->count();
	std::vector<int> outRows;
	outRows.resize(selected.count());
	for(int i=selected.count()-1; i>=0; i--)
	{
		int oldRow = m_orderListWidget->row(selected[i]);
		int newRow = oldRow;
		if (oldRow+1<lastOutRow)
		{
			newRow = oldRow+1;
		}
		lastOutRow = newRow;
		outRows[i] = newRow;
	}

	// also start from the highest row
	for(int i=selected.count()-1; i>=0; i--)
	{
		int index= m_orderListWidget->row(selected[i]);

		if(outRows[i]!=index)
		{
			dynamic_cast<HorizonFolderData*>(m_rep->data())->changeOrder(index,outRows[i]);

			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(outRows[i],item);
		}
	}

	// reset selection
	m_orderListWidget->clearSelection();
	for (int i=0;i<selected.count();i++)
	{
		selected[i]->setSelected(true);
	}
	connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	m_orderMutex.unlock();
}


void HorizonPropPanel::moveTop()
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();
	if (selected.size()>=m_orderListWidget->count())
	{
		return;
	}
	disconnect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	// reorder
	std::sort(selected.begin(), selected.end(), [this](const QListWidgetItem* first, const QListWidgetItem* second) {
		int rowFirst = this->m_orderListWidget->row(first);
		int rowSecond = this->m_orderListWidget->row(second);
		return rowFirst<rowSecond;
	});

	// start from the lowest row
	for(int i=0;i<selected.count();i++)
	{
		int index= m_orderListWidget->row(selected[i]);

		if(i!=index)
		{
			dynamic_cast<HorizonFolderData*>(m_rep->data())->changeOrder(index, i);

			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(i,item);
		}
	}

	// reset selection
	m_orderListWidget->clearSelection();
	for (int i=0;i<selected.count();i++)
	{
		selected[i]->setSelected(true);
	}
	connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	m_orderMutex.unlock();
}

void HorizonPropPanel::moveBottom()
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();
	if (selected.size()>=m_orderListWidget->count())
	{
		return;
	}
	disconnect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	// reorder
	std::sort(selected.begin(), selected.end(), [this](const QListWidgetItem* first, const QListWidgetItem* second)
	{
		int rowFirst = this->m_orderListWidget->row(first);
		int rowSecond = this->m_orderListWidget->row(second);
		return rowFirst<rowSecond;
	});

	// create the position post operation, since limit is m_orderListWidget->count, start from the highest row
	int nextOutRow = m_orderListWidget->count()-1;
	std::vector<int> outRows;
	outRows.resize(selected.count());
	for(int i=selected.count()-1; i>=0; i--)
	{
		outRows[i] = nextOutRow;
		nextOutRow--;
	}

	// also start from the highest row
	for(int i=selected.count()-1; i>=0; i--)
	{
		int index= m_orderListWidget->row(selected[i]);

		if(outRows[i]!=index)
		{
			dynamic_cast<HorizonFolderData*>(m_rep->data())->changeOrder(index,outRows[i]);

			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(outRows[i],item);
		}
	}

	// reset selection
	m_orderListWidget->clearSelection();
	for (int i=0;i<selected.count();i++)
	{
		selected[i]->setSelected(true);
	}
	connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	m_orderMutex.unlock();
}



void HorizonPropPanel::dataRemoved(IData* data)
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	QList<QListWidgetItem*> items =m_orderListWidget->findItems(data->name(),Qt::MatchExactly);
	if( items.size() >0) delete items[0];

	m_animSlider->setMaximum(m_orderListWidget->count() - 1 );

	m_orderMutex.unlock();
}

void HorizonPropPanel::dataAdded(IData* data)
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	m_orderListWidget->addItem(new QListWidgetItem(data->name()));

	m_animSlider->setMaximum(m_orderListWidget->count() - 1  );

	m_orderMutex.unlock();

	fillComboAttribut();
}


void HorizonPropPanel::setRangeToImage(QVector2D range)
{
	if(m_lastimage!= nullptr)
	{
		m_lastimage->setRange(0,range);
		m_lastimage->setRange(1,range);
		m_lastimage->setRange(2,range);
	}
}


void HorizonPropPanel::setRangeFromImage(unsigned int index,QVector2D range)
{
	m_palette->setRange(range);


}
void HorizonPropPanel::updatePalette(CUDARGBInterleavedImage* image )
{
	if(m_modeRGB == false)
	{
		if(image != nullptr) m_palette->setPaletteHolder( image->holder(0));

		/*if(image == m_lastimage)
			{
				qDebug()<<"image == lastimage  ";
				return;
			}*/
		if(image != nullptr)
		{

			//connect(m_palette, SIGNAL(rangeChanged(const QVector2D & )),this,SLOT(setRangeToImage(const QVector2D & )));
			//connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)), image ,SLOT(lookupTableChangedInternal(const LookupTable &)));
			connect(m_palette, SIGNAL(opacityChanged(float)),image, SLOT(setOpacity(float)));

			//connect(image,SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), this,SLOT(setRangeFromImage(unsigned int ,const QVector2D & )));
			//connect(image, SIGNAL(lookupTableChanged(const LookupTable &)), m_palette ,SLOT(lookupTableChangedInternal(const LookupTable &)));
			connect(image, SIGNAL(opacityChanged(float)),m_palette, SLOT(setOpacity(float)));
		}
		if(m_lastimage && m_palette)
		{
			//qDebug()<<"disconnect  "<<m_lastimage;
			///disconnect(m_palette, SIGNAL(rangeChanged(const QVector2D & )),this,SLOT(setRangeToImage(const QVector2D & )));
			disconnect(m_palette, SIGNAL(opacityChanged(float)),m_lastimage, SLOT(setOpacity(float)));

			//disconnect(m_lastimage,SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), this,SLOT(setRangeFromImage(unsigned int ,const QVector2D & )));
			disconnect(m_lastimage, SIGNAL(opacityChanged(float)),m_palette, SLOT(setOpacity(float)));
		}
		m_lastimage = image;

		return ;
	}

	if(image != nullptr)
	{
		m_paletteRGB->setPaletteHolder(0, image->holder(0));
		m_paletteRGB->setPaletteHolder(1, image->holder(1));

		m_paletteRGB->setPaletteHolder(2, image->holder(2));
	}



	if(image != nullptr)
	{
	//	m_palette->setPaletteHolder(0, image->holder(0));
		//m_palette->setPaletteHolder(1, image->holder(1));
		//m_palette->setPaletteHolder(2, image->holder(2));

	//	connect(m_paletteRGB, SIGNAL(lookupTableChanged(const LookupTable &)), image ,SLOT(lookupTableChangedInternal(const LookupTable &)));
		connect(m_paletteRGB, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),this,SLOT(setRangeLock(unsigned int ,const QVector2D & )));
		connect(m_paletteRGB, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),image,SLOT(setRange(unsigned int ,const QVector2D & )));
		connect(m_paletteRGB, SIGNAL(opacityChanged(float)),image, SLOT(setOpacity(float)));

		//connect(image, SIGNAL(lookupTableChanged(const LookupTable &)), m_paletteRGB ,SLOT(lookupTableChangedInternal(const LookupTable &)));
		connect(image,SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_paletteRGB,SLOT(setRange(unsigned int ,const QVector2D & )));
		connect(image, SIGNAL(opacityChanged(float)),m_paletteRGB, SLOT(setOpacity(float)));
	}

	if(image == m_lastimage)
	{
		return;
	}

	if(m_lastimage )
	{

		disconnect(m_paletteRGB, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),this,SLOT(setRangeLock(unsigned int ,const QVector2D & )));
		disconnect(m_paletteRGB, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),m_lastimage,SLOT(setRange(unsigned int ,const QVector2D & )));
		disconnect(m_paletteRGB, SIGNAL(opacityChanged(float)),m_lastimage, SLOT(setOpacity(float)));


		disconnect(m_lastimage,SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_paletteRGB,SLOT(setRange(unsigned int ,const QVector2D & )));
		disconnect(m_lastimage, SIGNAL(opacityChanged(float)),m_paletteRGB, SLOT(setOpacity(float)));
	}
	m_lastimage = image;
}

void HorizonPropPanel::warnBadAttribut(const QString& attributName) {
	QString errMessage = tr("Invalid attribut : ") + attributName;

	QList<FreeHorizon*> listFree =m_rep->horizonFolderData()->completOrderList();
	QStringList horizonsMissingAttribut;
	for(int i=0;i<listFree.count();i++)
	{
		bool notFound = true;
		int j=0;
		while(notFound && j<listFree[i]->m_attribut.size())
		{
			QString name = listFree[i]->m_attribut[j].name();
			notFound = attributName.compare(name)!=0;
			j++;
		}
		if (notFound)
		{
			horizonsMissingAttribut.push_back(listFree[i]->name());
		}
	}
	if (horizonsMissingAttribut.size()>0)
	{
		errMessage += tr(", see horizons : ");
		for (int i=0; i<horizonsMissingAttribut.size(); i++)
		{
			errMessage += "\n" + horizonsMissingAttribut[i];
		}
	}
	QMessageBox::warning(this, tr("Bad attribut"), errMessage);
}

void HorizonPropPanel::orderChangedFromData(int oldIndex, int newIndex)
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}

	disconnect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();

	QListWidgetItem* item = m_orderListWidget->takeItem(oldIndex);
	m_orderListWidget->insertItem(newIndex,item);

	// reset selection
	m_orderListWidget->clearSelection();
	for (int i=0;i<selected.count();i++)
	{
		selected[i]->setSelected(true);
	}
	connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));

	m_orderMutex.unlock();
}
