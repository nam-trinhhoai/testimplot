#include "palettewidget.h"
#include "histogramwidget.h"
#include "colortableregistry.h"
#include "colortablechooser.h"
#include "colortableselector.h"

#include <QLabel>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLineEdit>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSpacerItem>
#include <QLocale>
#include <QGroupBox>
#include <QComboBox>
#include <QStyledItemDelegate>
#include <QPainter>
#include <QApplication>
#include <QDebug>
#include "luteditdialog.h"
#include "qhistogram.h"
#include "ipaletteholder.h"
#include <iostream>
#include <cmath>

// comment it if you don't want modifs for this JIRA task
#define NEXTVSN_37

template<typename ... Args> struct SELECT {
	template<typename C, typename R>
	static constexpr auto OVERLOAD_OF(R (C::*pmf)(Args...)) -> decltype(pmf) {
		return pmf;
	}
};

PaletteWidget::PaletteWidget(QWidget* parent , Qt::WindowFlags f) :QWidget(parent,f) {
	m_editDialog=nullptr;
	QVBoxLayout *layout=new QVBoxLayout(this);
	//layout->setMargin(0);
	layout->setContentsMargins(0,0,0,0);
	m_histoWidget=new HistogramWidget(this);
	connect(m_histoWidget, SIGNAL(rangeChanged(const QVector2D &)), this, SLOT(histogramRangeChanged(const QVector2D &)));

	m_opacity=new QSlider(Qt::Orientation::Horizontal,this);
	m_opacity->setSingleStep(1);
	m_opacity->setTickInterval(10);
	m_opacity->setMinimum(0);
	m_opacity->setMaximum(100);
	m_opacity->setValue(100);

	QDoubleSpinBox* opacitySpin = new QDoubleSpinBox(this);
	opacitySpin->setMinimum(0);
	opacitySpin->setMaximum(1);
	opacitySpin->setSingleStep(0.01);
	opacitySpin->setDecimals(2);
	opacitySpin->setValue(1);
	connect(m_opacity, &QSlider::valueChanged,[=](int value){
		opacitySpin->setValue(0.01*value);
	});
	connect(opacitySpin, SELECT<double>::OVERLOAD_OF(&QDoubleSpinBox::valueChanged),[=](double value){
		m_opacity->setValue(value*100);
	});

	connect(m_opacity, SIGNAL(valueChanged(int)), this, SLOT(opacityChanged(int)));

	{
		QWidget *group = new QWidget(this);
		QHBoxLayout *ll = new QHBoxLayout(group);
		ll->setContentsMargins(5,5,0,0);
		group->setLayout(ll);
		QWidget *lutWidget = createLUTWidget();
		lutWidget->setStyleSheet("min-width: 150px;");
		lutWidget->setMinimumWidth(200);
		ll->addWidget(lutWidget, 1);
		m_edit=new QPushButton("Edit",this);
		m_edit->setFixedSize(40,24);
		m_edit->setDefault(false);
		m_edit->setAutoDefault(false);
		m_edit->setStyleSheet("min-width: 24px;");
		ll->addWidget(m_edit);
		ll->addSpacerItem(new QSpacerItem(0,0,QSizePolicy::Expanding, QSizePolicy::Minimum));

		layout->addWidget(group);
	}

#ifndef NEXTVSN_37
	connect(m_list, SIGNAL(currentIndexChanged(int)), this,SLOT(colorTableIndexChanged()));
#else
	connect(m_paletteSelector, SIGNAL(selectionChanged(ColorTable)),
	            this,SLOT(colorTableIndexChanged()));
#endif
	connect(m_edit, SIGNAL(clicked()), this,SLOT(colorTableEdited()));
	{
		QWidget * hWidget=new QWidget(this);
		QHBoxLayout *hBox=new QHBoxLayout(hWidget);
		//hBox->setMargin(0);
		hBox->setContentsMargins(0,0,0,0);
		hBox->addWidget(m_histoWidget, 1);

		QWidget * buttonWidget=new QWidget(this);
		QVBoxLayout *vBox=new QVBoxLayout(buttonWidget);
		//vBox->setMargin(0);
		vBox->setContentsMargins(0,0,0,0);
		m_reset=new QPushButton(this);
		m_reset->setFixedSize(24,24);
		m_reset->setStyleSheet("min-width: 24px;");

		m_reset->setDefault(false);
		m_reset->setAutoDefault(false);
		m_reset->setIcon(QIcon(":/palette/icons/undo.png"));
		vBox->addWidget(m_reset);
		connect(m_reset, SIGNAL(clicked()), this, SLOT(resetRange()));

		m_recompute=new QPushButton(this);
		m_recompute->setFixedSize(24,24);
		m_recompute->setStyleSheet("min-width: 24px;");
		m_recompute->setDefault(false);
		m_recompute->setAutoDefault(false);
		m_recompute->setIcon(QIcon(":/palette/icons/shrink.png"));
		vBox->addWidget(m_recompute);
		connect(m_recompute, SIGNAL(clicked()), this, SLOT(recompute()));


		m_wand=new QPushButton(this);
		m_wand->setFixedSize(24,24);
		m_wand->setStyleSheet("min-width: 24px;");
		m_wand->setDefault(false);
		m_wand->setAutoDefault(false);
		m_wand->setIcon(QIcon(":/palette/icons/wand.png"));
		vBox->addWidget(m_wand);
		connect(m_wand, SIGNAL(clicked()), this, SLOT(smartAdjust()));
		hBox->addWidget(buttonWidget,0,Qt::AlignmentFlag::AlignTop);

		layout->addWidget(hWidget);
	}
	{
		QWidget * rangeWidget=new QWidget(this);
		QHBoxLayout *hBox=new QHBoxLayout(rangeWidget);
		//hBox->setMargin(0);
		hBox->setContentsMargins(0,0,0,0);
		m_min=new QLineEdit(this);

		m_min->setLocale(QLocale::C);
		connect(m_min, SIGNAL(editingFinished()), this, SLOT(valueChanged()));
		hBox->addWidget(new QLabel("Min:",this));
		hBox->addWidget(m_min);

		m_max=new QLineEdit(this);
		m_max->setLocale(QLocale::C);
		connect(m_max, SIGNAL(editingFinished()), this, SLOT(valueChanged()));

		hBox->addSpacerItem(new QSpacerItem(0,0,QSizePolicy::Expanding, QSizePolicy::Minimum));
		hBox->addWidget(new QLabel("Max:",this));
		hBox->addWidget(m_max);

		layout->addWidget(rangeWidget,0,Qt::AlignmentFlag::AlignTop);
	}
	{
		QGroupBox * opacityWidget=new QGroupBox( "Opacity",this);

		QHBoxLayout *hBox=new QHBoxLayout(opacityWidget);
		//hBox->setMargin(0);
		hBox->setContentsMargins(0,0,0,0);
		hBox->addWidget(m_opacity);
		hBox->addWidget(opacitySpin);
		layout->addWidget(opacityWidget,0,Qt::AlignmentFlag::AlignTop);
	}

	layout->addSpacerItem(new QSpacerItem(0,0,QSizePolicy::Minimum, QSizePolicy::Expanding));
	setLookupTable(m_lookupTable);
}

void PaletteWidget::colorTableEdited()
{
	if(!m_image)return;
	QHistogram histo = m_image->computeHistogram(getRange(),QHistogram::HISTOGRAM_SIZE);
	if(!m_editDialog) {
		m_editDialog=new LUTEditDialog(histo, getRange(), m_lookupTable,this);
		connect(m_editDialog, SIGNAL(lookupTableChanged(const LookupTable &)), this,SLOT(lookupTableChangedInternal(const LookupTable &)));
	} else {
		m_editDialog->setHistogramAndLookupTable(histo, getRange(), m_lookupTable);
	}

	m_editDialog->show();

}

#if 0
void PaletteWidget::chooseColorTable()
{
        qDebug() << "PaletteWidget::chooseColorTable() called" << endl;
        QDialog* d = new ColorTableChooser(this);
        d->show();
}
#endif

void PaletteWidget::lookupTableChangedInternal(const LookupTable& colorTable)
{
	m_lookupTable=colorTable;
	updateColorTable();
	emit lookupTableChanged(colorTable);
}

void PaletteWidget::opacityChanged(int value)
{
	emit opacityChanged(value*0.01f);
}

void PaletteWidget::setPaletteHolder(IPaletteHolder *image)
{
	m_image=image;
	m_originalRange=m_image->dataRange();
	m_histo=m_image->computeHistogram(m_originalRange,QHistogram::HISTOGRAM_SIZE);
	updateHistogramAndColorTable();
	setRangeField(m_image->range());
}

void PaletteWidget::setRangeField(const QVector2D &r)
{
	m_min->setText(locale().toString(r.x(), 'f', 2));
	m_max->setText(locale().toString(r.y(), 'f', 2));
	m_histoWidget->setRange(r);
}


void PaletteWidget::resetRange()
{
	if(!m_image)return;

	setRangeField(m_originalRange);
	m_histoWidget->setRange(m_originalRange);
	m_histo = m_image->computeHistogram(m_originalRange,QHistogram::HISTOGRAM_SIZE);
	updateHistogramAndColorTable();
	emit rangeChanged(m_originalRange);
}

void PaletteWidget::smartAdjust()
{
	QVector2D adpatedRange=IPaletteHolder::smartAdjust(m_originalRange,m_histo);
	setRangeField(adpatedRange);

	emit rangeChanged(adpatedRange);
}

float PaletteWidget::getOpacity() const
{
        return m_opacity->value() * 0.01f;
}

QVector2D PaletteWidget::getRange() const
{
	QVector2D range;
	range.setX(locale().toDouble(m_min->text()));
	range.setY(locale().toDouble(m_max->text()));

	return range;
}

void PaletteWidget::recompute()
{
	if(!m_image)return;

	m_histo = m_image->computeHistogram(getRange(),QHistogram::HISTOGRAM_SIZE);
	updateHistogramAndColorTable();
}

void PaletteWidget::valueChanged()
{
	if(!m_image)return;

	QVector2D r=getRange();
	if ( r.x() > r.y()) {
		r.setX(r.y());
		m_min->setText(locale().toString(r.x()));
	}
	else if ( r.x() > r.y()) {
		r.setY(r.x());
		m_max->setText(locale().toString(r.y()));
	}
	m_histoWidget->setRange(r);
	emit rangeChanged(r);
}


void PaletteWidget::setOpacity(float val) {
	disconnect(m_opacity, SIGNAL(valueChanged(int)), this, SLOT(opacityChanged(int)));
	m_opacity->setValue((int)(val*100));
	connect(m_opacity, SIGNAL(valueChanged(int)), this, SLOT(opacityChanged(int)));
}

LookupTable PaletteWidget::getLookupTable()
{
	return m_lookupTable;
}

void PaletteWidget::setLookupTable(const LookupTable &table) {
	m_lookupTable=table;
#ifndef NEXTVSN_37
	disconnect(m_list, SIGNAL(currentIndexChanged(int)), this,SLOT(colorTableIndexChanged()));
	int pos = m_list->findData(table.getName().c_str(),Qt::DisplayRole);
	m_list->setCurrentIndex(pos);
	updateColorTable();
	connect(m_list, SIGNAL(currentIndexChanged(int )), this,SLOT(colorTableIndexChanged()));
#else
	// seems just to inhibate temporarily signal from selector?
	// QSignalBlocker instead?
	disconnect(m_paletteSelector, SIGNAL(selectionChanged(ColorTable)),
			this,SLOT(colorTableIndexChanged()));
	ColorTable ct = table.getColorTable();
	m_paletteSelector->setSelection(ct);
	connect(m_paletteSelector, SIGNAL(selectionChanged(ColorTable)),
			this,SLOT(colorTableIndexChanged()));
#endif
}

struct Delegate: public QStyledItemDelegate {
	Delegate(QObject* parent = nullptr) :
			QStyledItemDelegate(parent) {
	}
	void paint(QPainter* painter, const QStyleOptionViewItem& option,
			const QModelIndex& index) const {
		auto o = option;
		o.decorationSize.setWidth(80);
		o.decorationSize.setHeight(20);
		initStyleOption(&o, index);

		auto style = o.widget ? o.widget->style() : QApplication::style();
		style->drawControl(QStyle::CE_ItemViewItem, &o, painter, o.widget);
	}
	QSize sizeHint(const QStyleOptionViewItem &option,
			const QModelIndex &index) const Q_DECL_OVERRIDE
			{
		return QSize(80, 16);
	}
};

QWidget *PaletteWidget::createLUTWidget() {
#ifndef NEXTVSN_37
		m_list = new QComboBox(this);
		//m_list->setItemDelegate(new Delegate());
		//m_list->setSelectionMode(QAbstractItemView::SingleSelection);

		std::map<std::string, std::vector<ColorTable> >::const_iterator it =
				ColorTableRegistry::PALETTE_REGISTRY().getFamilies().begin();

		while (it != ColorTableRegistry::PALETTE_REGISTRY().getFamilies().end()) {

			std::vector<ColorTable> v = it->second;
			std::vector<ColorTable>::iterator itCol = v.begin();
			while (itCol != v.end()) {
				QImage img(itCol->size(), 16, QImage::Format_RGB32);
				QPainter p(&img);
				p.fillRect(img.rect(), Qt::black);

//				QRect rect = img.rect().adjusted(1, 1, -1, -1);
//				int* background = itCol->getBackground();
//				p.fillRect(rect,
//						QColor(background[0], background[1], background[2], 0));
				for (int i = 0; i < itCol->size(); i++) {
					QRect rect = img.rect().adjusted(i, 1, i + 1, 14);
					const std::array<int, 4> color = itCol->getColors(i);
					p.fillRect(rect,QColor(color[0], color[1], color[2], color[3]));
				}

				m_list->addItem(QPixmap::fromImage(img.scaled(80, 16)),
						tr(itCol->getName().c_str()), QVariant::fromValue(*itCol));
				itCol++;
			}
			it++;
		}
		return m_list;
#else
		m_paletteSelector = new ColorTableSelector(this);
		return m_paletteSelector;
#endif
}

void PaletteWidget::updateHistogramAndColorTable()
{
	m_histoWidget->setHistogramAndLookupTable(m_histo, getRange(), m_lookupTable);
	if(m_editDialog){
		m_editDialog->setHistogramAndLookupTable( m_histo, getRange(), m_lookupTable );
	}
}

void PaletteWidget::updateColorTable()
{
	m_histoWidget->setHistogramAndLookupTable(m_histo, getRange(), m_lookupTable);
	if(m_editDialog){
		m_editDialog->setLookupTable( m_lookupTable );
	}
}

void PaletteWidget::colorTableIndexChanged() {
	if(!m_image)return;

#ifndef NEXTVSN_37
	ColorTable currentTable= m_list->currentData(Qt::UserRole).value<ColorTable>();
	m_lookupTable = LookupTable(currentTable);
#else
	m_lookupTable = LookupTable(m_paletteSelector->getCurrentSelection());
#endif
	updateColorTable();

	emit lookupTableChanged(m_lookupTable);
}

void PaletteWidget::histogramRangeChanged(const QVector2D & range)
{
	setRangeField(range);
	emit rangeChanged(range);
}

PaletteWidget::~PaletteWidget() {
	// TODO Auto-generated destructor stub
}


void PaletteWidget::setRange(const QVector2D& range)
{
    m_min->setText(locale().toString(range.x()));
    m_max->setText(locale().toString(range.y()));

    m_histoWidget->setRange(range);
}

