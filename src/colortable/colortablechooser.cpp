#include "colortablechooser.h"
#include "colortableselector.h"

#include "colortableregistry.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPushButton>
#include <QRect>
#include <QSignalBlocker>
#include <QSizePolicy>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QVBoxLayout>
#include <QVariant>

#include <QDebug>

ColorTableChooser::ColorTableChooser(ColorTableSelector* from,
		QWidget* parent) : QDialog(parent)
{
	m_from = from;
	m_filterData = nullptr;

	QVBoxLayout *mainLayout=new QVBoxLayout(this);
	mainLayout->setSpacing(0);

	m_filterData = new QLineEdit;
	mainLayout->addWidget(m_filterData);

	m_table = new QTableWidget(getNTotalRows(), 2);
	m_table->setEditTriggers(QAbstractItemView::CurrentChanged);
	m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
	m_table->setSelectionMode(QAbstractItemView::SingleSelection);
	m_table->setHorizontalHeaderLabels({"Name", "Family"});
	mainLayout->addWidget(m_table);
	int nRows = populateTableWidget();

	//m_table->setRowCount(nRows);
	m_table->verticalHeader()->setVisible(false);
	m_table->resizeColumnsToContents();
	m_table->setSortingEnabled(true);

	m_idxFilter=0;	// "none" by default

#if 0
	QSizePolicy horFixed(QSizePolicy::Fixed, QSizePolicy::Expanding);
	QWidget* lateralWidget = new QWidget(this);
	lateralWidget->setSizePolicy(horFixed);
	QVBoxLayout *lateralLayout = new QVBoxLayout(lateralWidget);
	QWidget* filterW= new QWidget(lateralWidget);
	QHBoxLayout* filterLayout = new QHBoxLayout(filterW);
	QLabel* lab = new QLabel("filter");
	filterLayout->addWidget(lab);
	m_filterType = new QComboBox();
	m_filterType->addItems({"none","by family","with size lower than", "with size greater than"});
	m_idxFilter=0;	// "none" by default
	filterLayout->addWidget(m_filterType);
	lateralLayout->addWidget(filterW);

	m_filterData = new QLineEdit(lateralWidget);
	lateralLayout->addWidget(m_filterData);
#if 0
	m_filterApply = new QPushButton("Apply");
	lateralLayout->addWidget(m_filterApply);
#else
	QWidget* applyW= new QWidget(lateralWidget);
	QHBoxLayout* applyLayout = new QHBoxLayout(applyW);
	m_filterApply = new QPushButton("Apply");
	applyLayout->addWidget(m_filterApply);
	applyLayout->addStretch(1);
	lateralLayout->addWidget(applyW);
#endif
	lateralLayout->addStretch(1);

	mainLayout->addWidget(lateralWidget);
#endif

#if 0
    connect( m_filterApply, &QPushButton::clicked,
             this, &ColorTableChooser::applyFilter );
    connect(m_filterType, SIGNAL(currentIndexChanged(int )),
    		this, SLOT(filterChanged(int)));
#endif
    connect(m_table, &QTableWidget::itemSelectionChanged,
                 this, &ColorTableChooser::ciChange);
    connect(m_filterData , &QLineEdit::textChanged,
               this, &ColorTableChooser::applyFilter );

	setFixedWidth(sizeHint().width()+30);
	resize(sizeHint().width()+30,600);
}

ColorTableChooser::~ColorTableChooser()
{
}

void ColorTableChooser::selectLineFromData(const ColorTable& hdToSelect)
{
    //qDebug() << "ColorTableChooser::selectLineFromData() called with " << hdToSelect->getValue();
    int iRow, nRows = m_table->rowCount();

    for(iRow= 0; iRow < nRows; iRow++) {
        QVariant hdVar = m_table->item( iRow,0)->data( Qt::UserRole );

        if( hdVar.canConvert<ColorTable>() ) {
            ColorTable hd = qvariant_cast<ColorTable>(hdVar);
            if(hd==hdToSelect)
                break;
        }
    }

    if(iRow == nRows) {
        qDebug() << "ColorTableChooser::selectLineFromData(): unexpected case";
        return;
    }

#if 0
    QTableWidgetSelectionRange rangeSelOld(m_curRow, 0, m_curRow, 1);
    QTableWidgetSelectionRange rangeSelNew(iRow, 0, iRow, 1);
#endif
    m_curRow= iRow;
    {
        const QSignalBlocker blocker(m_table);
#if 0
        if( m_curRow != -1) {
            m_table->setRangeSelected(rangeSelOld, false);
        }
        m_table->setRangeSelected(rangeSelNew, true);
#endif
        m_table->selectRow(iRow);
    }
}

void ColorTableChooser::applyFilter()
{
    qDebug() << "ColorTableChooser::applyFilter() called";
    // m_table->setRowCount(0);	// clears elements
    // m_table->setRowCount(getNTotalRows());
    m_table->setSortingEnabled(false);
    populateTableWidget();
    m_table->setSortingEnabled(true);
}

void ColorTableChooser::filterChanged(int idxNew)
{
	qDebug() << "ColorTableChooser::filterChanged() called with index" << idxNew;
	m_idxFilter = idxNew;
}

void ColorTableChooser::ciChange()
{
	QList<QTableWidgetItem*> selectedItems = m_table->selectedItems();
    if(selectedItems.size()==0 || selectedItems[0]->row() == m_curRow) {
        return;
    }
    QTableWidgetItem* item = selectedItems[0];

    m_curRow=item->row();
    QVariant d = m_table->item(m_curRow,0)->data(/*QTableWidgetItem::Type*/Qt::UserRole );
    //qDebug() << "QVariant data is " << d.toString();

    if( d.canConvert<ColorTable>() ) {
		ColorTable hd = qvariant_cast<ColorTable>(d);
		m_from->changeSelection(hd);
	}
}

bool ColorTableChooser::filterColorTable(const ColorTable& col, const std::string& family) {
	QString filterText;
	if (m_filterData) {
		filterText = m_filterData->text();
	}
	bool useFilter = m_filterData!=nullptr && !filterText.isNull() && !filterText.isEmpty();

	bool valid = !useFilter;
	if (!valid && m_idxFilter==0) {
		 valid = QString::fromStdString(col.getName()).contains(filterText, Qt::CaseInsensitive);
	} else if (!valid && m_idxFilter==1) {
		valid = QString::fromStdString(family).contains(filterText, Qt::CaseInsensitive);
	}
	return valid;
}

int ColorTableChooser::getNTotalRows()
{
	static int nTotRows = -1;

	if(nTotRows == -1) {
		std::map<std::string, std::vector<ColorTable> >::const_iterator it =
				ColorTableRegistry::PALETTE_REGISTRY().getFamilies().begin();

		nTotRows= 0;

		while (it != ColorTableRegistry::PALETTE_REGISTRY().getFamilies().end()) {

			std::vector<ColorTable> v = it->second;
			std::vector<ColorTable>::iterator itCol = v.begin();
			while (itCol != v.end()) {
				itCol++;
				++nTotRows;
			}

			it++;
		}
	}

	return nTotRows;
}

int ColorTableChooser::populateTableWidget() {
		m_table->clearContents();
		m_table->setRowCount(getNTotalRows());

		std::map<std::string, std::vector<ColorTable> >::const_iterator it =
		ColorTableRegistry::PALETTE_REGISTRY().getFamilies().begin();

		int iRow= 0;

		while (it != ColorTableRegistry::PALETTE_REGISTRY().getFamilies().end()) {

			std::vector<ColorTable> v = it->second;
			std::vector<ColorTable>::iterator itCol = v.begin();
			while (itCol != v.end()) {
				if (filterColorTable(*itCol, it->first)) {
					QImage img(itCol->size(), 32, QImage::Format_RGB32);
					QPainter p(&img);
					p.fillRect(img.rect(), Qt::black);


					for (int i = 0; i < itCol->size(); i++) {
						QRect rect = img.rect().adjusted(i, 1, i + 1, 30);
						const std::array<int, 4> color = itCol->getColors(i);
						p.fillRect(rect,QColor(color[0], color[1], color[2], color[3]));
					}

#if 0
					m_list->addItem(QPixmap::fromImage(img.scaled(80, 16)),
							tr(itCol->getName().c_str()), QVariant::fromValue(*itCol));
#else
					QTableWidgetItem *item0 = new QTableWidgetItem(
							QPixmap::fromImage(img.scaled(80, 32)),
							QTableWidget::tr(itCol->getName().c_str()) );
					QTableWidgetItem *item1 = new QTableWidgetItem(
							QTableWidget::tr(it->first.c_str() ) );
					item0->setData( Qt::UserRole, QVariant::fromValue(*itCol));
					item0->setFlags(item0->flags() & ~Qt::ItemIsEditable);
					item1->setData( Qt::UserRole, QVariant::fromValue(*itCol));
					item1->setFlags(item1->flags() & ~Qt::ItemIsEditable);
					m_table->setItem(iRow, 0, item0);
					m_table->setItem(iRow, 1, item1);
					iRow++;
#endif
				}
				itCol++;
			}

			it++;
		}

		m_table->setRowCount(iRow);

		return iRow;
	}


