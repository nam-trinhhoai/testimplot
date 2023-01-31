#include "horizonfolderproppanelonrandom.h"


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
#include "abstractinnerview.h"
#include "pointpickingtask.h"
#include "idata.h"
#include "horizonfolderdata.h"
//#include "fixedrgblayersfromdatasetandcube.h"
#include "orderstackhorizonwidget.h"
//#include "rgbpalettewidget.h"
#include "freehorizon.h"
#include "horizonfolderreponrandom.h"

HorizonFolderPropPanelOnRandom::HorizonFolderPropPanelOnRandom(HorizonFolderRepOnRandom *rep,  QWidget *parent) :
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
			m_orderListWidget->addItem(new QListWidgetItem(fixed->name()));

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
		m_comboAttribut->addItems(getAttributesAvailable());
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
		lay1->addWidget(m_animSlider);
		lay1->addWidget(m_playButton);

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



		mainLayout01->addLayout(mainLayout02);
		mainLayout01->addLayout(lay2);
		mainLayout01->addLayout(lay0);
		mainLayout01->addLayout(lay1);


		//Palettes
		/*m_paletteRGB = new RGBPaletteWidget(this);
		m_paletteRGB->hide();
		mainLayout01->addWidget(m_paletteRGB);


		m_palette = new PaletteWidget(this);
		m_palette->hide();
		mainLayout01->addWidget(m_palette);*/
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

		connect(m_speedSlider, SIGNAL(valueChanged(int)),this,SLOT(speedChanged(int)));

		//connect(m_comboView3D,SIGNAL(currentIndexChanged(int)),this,SLOT(view3DChanged(int)));
		connect(m_comboAttribut,SIGNAL(currentIndexChanged(int)),this,SLOT(attributChanged(int)));



		connect(m_rep->data(),SIGNAL(layerAdded(IData*)),this,SLOT(dataAdded(IData*)));
		connect(m_rep->data(),SIGNAL(layerRemoved(IData*)),this,SLOT(dataRemoved(IData*)));
		connect(m_rep->data(),SIGNAL(orderChanged(int, int)), this, SLOT(orderChangedFromData(int ,int)));


		int indexDefault = m_comboAttribut->findText("spectrum",Qt::MatchContains);
		if(indexDefault>=0 )
		{
			m_comboAttribut->setCurrentIndex(indexDefault);

			m_animSlider->setValue(0);
		}

}
HorizonFolderPropPanelOnRandom::~HorizonFolderPropPanelOnRandom() {

}

QStringList HorizonFolderPropPanelOnRandom::getAttributesAvailable()
{
	QList<FreeHorizon*> listFree =m_rep->horizonFolderData()->completOrderList();
	QStringList listAttributs;
/*	if(listFree.count()>0)
	{
		for(int j=0;j< listFree[0]->m_attribut.size();j++)
		{
			listAttributs.push_back(listFree[0]->m_attribut[j].name());
		}
	}*/


	for(int i=1;i<listFree.count();i++)
	{
		for(int j=0;j< listFree[i]->m_attribut.size();j++)
		{
			QString name = listFree[i]->m_attribut[j].name();
			if(name!="" && ! listAttributs.contains(name))
			{

				listAttributs.append(name);
			}
		}
	}
	return listAttributs;
}


void HorizonFolderPropPanelOnRandom::attributChanged(int i)
{
	if(i>=0)
	{
		/*if(m_comboAttribut->currentText().contains("spectrum") )
			m_modeRGB = true;
		else
			m_modeRGB= false;*/

	//	m_rep->setNameAttribut(m_comboAttribut->currentText());



	/*	if(m_modeRGB)
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
		}*/


		dynamic_cast<HorizonFolderData*>(m_rep->data())->setCurrentData(m_animSlider->value());

	//	m_lastRGB = m_modeRGB;
	}

}

void HorizonFolderPropPanelOnRandom::add()
{

}


void HorizonFolderPropPanelOnRandom::remove()
{

}

void HorizonFolderPropPanelOnRandom::playAnimation(bool actif)
{
	if(actif)
	{
		m_animTimer->start(m_speedAnim);
	}
	else
	{
		m_animTimer->stop();
	}
}
void HorizonFolderPropPanelOnRandom::updateAnimation()
{
	int index = m_animSlider->value();
	int newIndex = index-1;
	if(newIndex <0) newIndex = m_animSlider->maximum();
	m_animSlider->setValue(newIndex);
}

void HorizonFolderPropPanelOnRandom::speedChanged(int value)
{
	m_speedAnim =m_speedSlider->maximum()+ m_speedSlider->minimum() - value;
	if(m_animTimer->isActive())
	{
		m_animTimer->setInterval(m_speedAnim);
	}
}

void HorizonFolderPropPanelOnRandom::trt_basketListSelectionChanged()
{
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();
	if (selected.size()==0)
	{
		return;
	}
	int index= m_orderListWidget->row(selected[0]);

	if(index>=0 && index !=  m_animSlider->value())
	{
		//QSignalBlocker b1(m_animSlider);
		m_animSlider->setValue(index);
		moveAnimation(index);
		//m_lastSelected = index;
	}
}

void HorizonFolderPropPanelOnRandom::moveAnimation(int value)
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

void HorizonFolderPropPanelOnRandom::moveUp()
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

void HorizonFolderPropPanelOnRandom::moveDown()
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


void HorizonFolderPropPanelOnRandom::moveTop()
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

void HorizonFolderPropPanelOnRandom::moveBottom()
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



void HorizonFolderPropPanelOnRandom::dataRemoved(IData* data)
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

void HorizonFolderPropPanelOnRandom::dataAdded(IData* data)
{
	if (!m_orderMutex.tryLock())
	{
		return;
	}
	//qDebug()<<" dataAdded ....";

	m_orderListWidget->addItem(new QListWidgetItem(data->name()));

	m_animSlider->setMaximum(m_orderListWidget->count() - 1  );

	m_orderMutex.unlock();

	m_comboAttribut->clear();
	m_comboAttribut->addItems(getAttributesAvailable());

}

void HorizonFolderPropPanelOnRandom::orderChangedFromData(int oldIndex, int newIndex)
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


