
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QToolButton>
#include <QComboBox>
#include <QLabel>
#include <QTimer>
#include "horizonfolderdata.h"
#include "orderstackhorizonwidget.h"
#include "fixedrgblayersfromdatasetandcube.h"

OrderStackHorizonWidget::OrderStackHorizonWidget(GeotimeGraphicsView* graphicsview,WorkingSetManager* workingset,QWidget* parent):QWidget(parent)
{
	m_workingset = workingset;


	connect(m_workingset->folders().horizonsFree,SIGNAL(dataAdded(IData*)),this, SLOT(dataAdded(IData*)));
	connect(graphicsview,SIGNAL(registerAddView3D()),this, SLOT(addView3D()));

	connect(m_workingset->folders().horizonsFree,SIGNAL(dataRemoved(IData*)),this, SLOT(dataRemoved(IData*)));
	m_graphicsview = graphicsview;
	QList<IData*> listData = m_workingset->folders().horizonsFree->data();


	m_animTimer = new QTimer(this);
	connect(m_animTimer, &QTimer::timeout, this, &OrderStackHorizonWidget::updateAnimation);


	QVBoxLayout * mainLayout01 = new QVBoxLayout(this);
//	m_filterEdit = new QLineEdit;
		QHBoxLayout * mainLayout02 = new QHBoxLayout;
	//	m_listItemToSelectWidget = new QListWidget;
	//	m_listItemToSelectWidget->setSelectionMode(QAbstractItemView::MultiSelection);

		m_orderListWidget = new QListWidget;
		int nbelem = listData.size();
		//qDebug()<<" nbelem :" <<nbelem;

		for(int i=0;i<listData.size();i++)
		{
			FixedRGBLayersFromDatasetAndCube* fixed = dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(listData[i]);
			//qDebug()<<" ==> "<<fixed->getName();

			m_dataOrderList.push_back(fixed);
			m_orderListWidget->addItem(new QListWidgetItem(fixed->getName()));
			m_indexList.push_back(i);
		}

		qDebug()<<" m_indexList :" <<m_indexList;

		/*for(int i=0;i<10 ;i++)
		{
			QString nom = "item"+QString::number(i);
			m_orderListWidget->addItem(new QListWidgetItem(nom));
		}*/
		//m_orderListWidget->setSelectionMode(QAbstractItemView::MultiSelection);
	//	mainLayout02->addWidget(m_listItemToSelectWidget);
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
		QLabel* attributelabel = new QLabel("View/attrib");
		m_comboView3D = new QComboBox();
		m_comboAttribut = new QComboBox();
		m_comboAttribut->addItems(getAttributesAvailable());
		lay0->addWidget(attributelabel);
		lay0->addWidget(m_comboView3D);
		lay0->addWidget(m_comboAttribut);


		QHBoxLayout* lay1 = new QHBoxLayout();
		m_animSlider = new QSlider(Qt::Orientation::Horizontal);
		m_animSlider->setMinimum(0);
		m_animSlider->setMaximum(nbelem-1);
		m_animSlider->setValue(0);
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


		//pushbutton_databaseUpdate = new QPushButton("DataBase Update");

		//labelSearchHelp = new QLabel("");
		//labelSearchHelp->setVisible(false);

		//mainLayout01->addWidget(labelSearchHelp);
	//	mainLayout01->addWidget(m_filterEdit);
		mainLayout01->addLayout(mainLayout02);
		mainLayout01->addLayout(lay2);
		mainLayout01->addLayout(lay0);
		mainLayout01->addLayout(lay1);

		//setContextMenu(true);

	//	connect(m_filterEdit, SIGNAL(textChanged(QString)), this, SLOT(filterChanged(QString)));
	  //  connect(m_orderListWidget, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(trt_basketListClick(QListWidgetItem*)));
	    connect(m_orderListWidget, SIGNAL(itemSelectionChanged()), this, SLOT(trt_basketListSelectionChanged()));


	    connect(pushbutton_up, SIGNAL(clicked()), this, SLOT(moveUp()));
	    connect(pushbutton_down, SIGNAL(clicked()), this, SLOT(moveDown()));
	    connect(pushbutton_top, SIGNAL(clicked()), this, SLOT(moveTop()));
	    connect(pushbutton_bottom, SIGNAL(clicked()), this, SLOT(moveBottom()));


	    connect(m_animSlider, SIGNAL(valueChanged(int)),this,SLOT(moveAnimation(int)));
	    connect(m_animSlider, SIGNAL(sliderMoved(int)),this,SLOT(moveAnimation(int)));
	    connect(m_playButton, SIGNAL(toggled(bool)),this,SLOT(playAnimation(bool)));

	    connect(m_speedSlider, SIGNAL(valueChanged(int)),this,SLOT(speedChanged(int)));

	    connect(m_comboView3D,SIGNAL(currentIndexChanged(int)),this,SLOT(view3DChanged(int)));
	    connect(m_comboAttribut,SIGNAL(currentIndexChanged(int)),this,SLOT(attributChanged(int)));




	    int index = 0;
	    int index3D = 1;
	    while(index<m_graphicsview->getInnerViews().size()) {//}&& dynamic_cast<ViewQt3D*>(m_graphicsview->getInnerViews()[index])==nullptr) {

	    	ViewQt3D* view3D = dynamic_cast<ViewQt3D*>(m_graphicsview->getInnerViews()[index]);
	    	if(view3D !=nullptr)
	    	{
	    		m_comboView3D->addItem("View3D "+QString::number(m_index3D));
	    		m_viewIndex.push_back(index);
	    		m_viewAttribIndex.push_back(qMakePair(m_index3D,0));
	    		m_index3D++;
	    		  qDebug()<<"-------------> view index :"<<index;
	    	}
	    	index++;
	    }

	   // qDebug()<<" view index :"<<m_viewIndex[0];
	   // m_viewIndex =index;


	//	connect(pushbutton_add, SIGNAL(clicked()), this, SLOT(add()));
	//	connect(pushbutton_sub, SIGNAL(clicked()), this, SLOT(remove()));
		//connect(pushbutton_databaseUpdate, SIGNAL(clicked()), this, SLOT(trt_dataBaseUpdate()));

		//connect(m_listItemToSelectWidget, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(ProvideContextMenuList(const QPoint &)));
		//connect(m_orderListWidget, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(ProvideContextMenubasket(const QPoint &)));

}

OrderStackHorizonWidget::~OrderStackHorizonWidget()
{
}

QStringList OrderStackHorizonWidget::getAttributesAvailable()
{
	QStringList res= {"Iso","Spectrum","Mean","GCC"};
	return res;
}


void OrderStackHorizonWidget::view3DChanged(int i)
{
	//qDebug()<<"view3DChanged  "<<i;
	if(i< m_viewAttribIndex.count())
	{
		m_comboAttribut->setCurrentIndex(m_viewAttribIndex[i].second );
	}
}

void OrderStackHorizonWidget::attributChanged(int i)
{
	int index = m_comboView3D->currentIndex();
	m_viewAttribIndex[index].second = i;

	//qDebug()<<" view 3D : "<<index<<" , attribut :"<<m_comboAttribut->currentText();
}

void OrderStackHorizonWidget::addView3D()
{
	int index=m_graphicsview->getInnerViews().size()-1;

	qDebug()<<" ADD VIEW 3D "<<index;

	m_comboView3D->addItem("View3D "+QString::number(m_index3D));
	m_viewIndex.push_back(index);
	m_viewAttribIndex.push_back(qMakePair(m_index3D,0));
	m_index3D++;
}

void OrderStackHorizonWidget::add()
{

}


void OrderStackHorizonWidget::remove()
{

}

void OrderStackHorizonWidget::playAnimation(bool actif)
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
void OrderStackHorizonWidget::updateAnimation()
{
	int index = m_animSlider->value();
	int newIndex = index+1;
	if(newIndex > m_animSlider->maximum()) newIndex = 0;
	m_animSlider->setValue(newIndex);
}

void OrderStackHorizonWidget::speedChanged(int value)
{
	m_speedAnim =m_speedSlider->maximum()+ m_speedSlider->minimum() - value;
	if(m_animTimer->isActive())
	{
		m_animTimer->setInterval(m_speedAnim);
	}
}

void OrderStackHorizonWidget::trt_basketListSelectionChanged()
{
	int index = m_orderListWidget->currentRow();

	if(index>=0)
	{
		QSignalBlocker b1(m_animSlider);
		m_animSlider->setValue(index);

		m_lastSelected = index;
	}
}

void OrderStackHorizonWidget::moveAnimation(int value)
{

	qDebug()<<" move animation"<<value;
	if( value >=0 && value< m_orderListWidget->count())
	{
		int index = m_indexList[value];
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
		}


		m_lastSelected = value;
	}
}

void OrderStackHorizonWidget::moveUp()
{
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();

	for(int i=0;i<selected.count();i++)
	{
		int index= m_orderListWidget->row(selected[i]);

		if(index >0)
		{
			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(index-1,item);
			m_orderListWidget->setCurrentItem(item);
			int lastindex = m_indexList[index];
			m_indexList[index] = m_indexList[index-1];
			m_indexList[index-1] = lastindex;
		}
	}
}

void OrderStackHorizonWidget::moveDown()
{
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();

	for(int i=0;i<selected.count();i++)
	{
		int index= m_orderListWidget->row(selected[i]);
		//qDebug()<<"index :"<<index<<" ,count :"<<m_orderListWidget->count();
		if(index <m_orderListWidget->count()-1)
		{
			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(index+1,item);
			m_orderListWidget->setCurrentItem(item);
			int lastindex = m_indexList[index];
			m_indexList[index] = m_indexList[index+1];
			m_indexList[index+1] = lastindex;
		}
	}
}


void OrderStackHorizonWidget::moveTop()
{
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();


	for(int i=0;i<selected.count();i++)
	{
		int index= m_orderListWidget->row(selected[i]);

		if(index >0)
		{
			QListWidgetItem* item = m_orderListWidget->takeItem(index);
			m_orderListWidget->insertItem(i,item);
			m_orderListWidget->setCurrentItem(item);

			int lastindex = m_indexList[0];
			m_indexList[0] = m_indexList[index];

			for(int i=index;i>1;i--)
				m_indexList[i] = m_indexList[i-1];

			m_indexList[1] =lastindex;
		}
	}
}

void OrderStackHorizonWidget::moveBottom()
{
	QList<QListWidgetItem*> selected = m_orderListWidget->selectedItems();

		for(int i=0;i<selected.count();i++)
		{
			int index= m_orderListWidget->row(selected[i]);
			//qDebug()<<"index :"<<index<<" ,count :"<<m_orderListWidget->count();
			if(index <m_orderListWidget->count()-1)
			{
				QListWidgetItem* item = m_orderListWidget->takeItem(index);
				m_orderListWidget->addItem(item);
				m_orderListWidget->setCurrentItem(item);

				int last = m_orderListWidget->count()-1;
				int lastindex = m_indexList[last];
				m_indexList[last] = m_indexList[index];

				for(int i=index;i<last-1;i++)
					m_indexList[i] = m_indexList[i+1];

				m_indexList[last] =lastindex;
			}
		}
}

void OrderStackHorizonWidget::apply()
{

}

void OrderStackHorizonWidget::dataRemoved(IData* data)
{
	qDebug()<<"TODO dataRemoved ....";
}

void OrderStackHorizonWidget::dataAdded(IData* data)
{

	FixedRGBLayersFromDatasetAndCube* fixed = dynamic_cast<FixedRGBLayersFromDatasetAndCube*>(data);
	m_dataOrderList.push_back(fixed);
	m_orderListWidget->addItem(new QListWidgetItem(fixed->getName()));
	m_indexList.push_back(m_indexList.count());


	int nbelem = m_dataOrderList.size()-1;

	qDebug()<<"dataAdded  ...."<<m_indexList;
	m_animSlider->setMaximum(nbelem-1);

}










