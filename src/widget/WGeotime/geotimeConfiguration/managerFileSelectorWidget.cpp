
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSplitter>
#include <QPushButton>
#include <QWidget>
#include <QSplitter>
#include <QDialogButtonBox>
#include <QTreeWidget>
#include <seismicinformation.h>
#include <loginformation.h>

#include <seismicinformationaggregator.h>
#include <nextvisionhorizoninformation.h>
#include <nextvisionhorizoninformationaggregator.h>
#include <loginformationaggregator.h>
#include <managerwidget.h>
#include <managerFileSelectorWidget.h>



ManagerFileSelectorWidget::ManagerFileSelectorWidget(IInformationAggregator* aggregator, QWidget* parent)
{
	m_aggregator = aggregator;
	m_aggregator->setParent(this);

	setMinimumSize(700,600);
	// setAttribute(Qt::WA_DeleteOnClose);
	SeismicInformationAggregator *agg = dynamic_cast<SeismicInformationAggregator*>(aggregator);
	if ( agg )
	{
		setWindowTitle("Nextvision: " + agg->projectName() + " - " + agg->surveyName());
	}
	else
		setWindowTitle("Nextvision");

	QVBoxLayout* mainLayout = new QVBoxLayout;
	m_treeWidget = new TreeWidget(m_aggregator);
	m_treeWidget->setSelectButtonVisible(false);

	QSplitter *splitter = new QSplitter(Qt::Horizontal);
	splitter->setStretchFactor(1,3);
	splitter->addWidget(m_treeWidget);
	mainLayout->addWidget(splitter);
	QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	mainLayout->addWidget(buttonBox);
	setLayout(mainLayout);
	connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
}


ManagerFileSelectorWidget::~ManagerFileSelectorWidget()
{

}

std::pair<std::vector<QString>, std::vector<QString>> ManagerFileSelectorWidget::seismicGetSelectedNames() const
{
	std::pair<std::vector<QString>, std::vector<QString>> out;
	std::vector<QString> names;
	std::vector<QString> paths;
	QTreeWidget *treeWidget = m_treeWidget->getTreeWidget();
	SeismicInformationAggregator *agg = dynamic_cast<SeismicInformationAggregator*>(m_aggregator);
	if ( !agg ) return out;
	for( int i = 0; i < treeWidget->topLevelItemCount(); ++i )
	{
		QTreeWidgetItem *item = treeWidget->topLevelItem(i);
		if ( item->checkState(0) )
		{
			QString name0 = item->text(0);
			for (int j=0; j<agg->size(); j++)
			{
				if ( agg->at(j)->name() == name0 )
				{
					names.push_back(name0);
					paths.push_back(agg->at(j)->mainPath());
				}
			}
		}
	}
	out = std::make_pair(names, paths);
	return out;
}

std::pair<std::vector<QString>, std::vector<QString>> ManagerFileSelectorWidget::nextvisionHorizonGetSelectedNames() const
{
	std::pair<std::vector<QString>, std::vector<QString>> out;
	std::vector<QString> names;
	std::vector<QString> paths;
	QTreeWidget *treeWidget = m_treeWidget->getTreeWidget();
	NextvisionHorizonInformationAggregator *agg = dynamic_cast<NextvisionHorizonInformationAggregator*>(m_aggregator);
	if ( !agg ) return out;
	for( int i = 0; i < treeWidget->topLevelItemCount(); ++i )
	{
		QTreeWidgetItem *item = treeWidget->topLevelItem(i);
		if ( item->checkState(0) )
		{
			QString name0 = item->text(0);
			for (int j=0; j<agg->size(); j++)
			{
				if ( agg->at(j)->name() == name0 )
				{
					names.push_back(name0);
					paths.push_back(agg->at(j)->mainPath());
				}
			}
		}
	}
	out = std::make_pair(names, paths);
	return out;
}

std::pair<std::vector<QString>, std::vector<QString>> ManagerFileSelectorWidget::logGetSelectedNames() const
{
	std::pair<std::vector<QString>, std::vector<QString>> out;
	std::vector<QString> names;
	std::vector<QString> paths;
	QTreeWidget *treeWidget = m_treeWidget->getTreeWidget();
	LogInformationAggregator *agg = dynamic_cast<LogInformationAggregator*>(m_aggregator);
	if ( !agg ) return out;
	for( int i = 0; i < treeWidget->topLevelItemCount(); ++i )
	{
		QTreeWidgetItem *item = treeWidget->topLevelItem(i);
		if ( item->checkState(0) )
		{
			QString name0 = item->text(0);
			for (int j=0; j<agg->size(); j++)
			{
				if ( agg->at(j)->name() == name0 )
				{
					names.push_back(name0);
					paths.push_back(agg->at(j)->mainPath());
				}
			}
		}
	}
	out = std::make_pair(names, paths);
	return out;
}


 std::pair<std::vector<QString>, std::vector<QString>> ManagerFileSelectorWidget::getSelectedNames() const{
	 std::pair<std::vector<QString>, std::vector<QString>> out;

	SeismicInformationAggregator *aggS = dynamic_cast<SeismicInformationAggregator*>(m_aggregator);
	if ( aggS ) return seismicGetSelectedNames();

	NextvisionHorizonInformationAggregator *aggH = dynamic_cast<NextvisionHorizonInformationAggregator*>(m_aggregator);
	if ( aggH ) return nextvisionHorizonGetSelectedNames();

	LogInformationAggregator *aggL = dynamic_cast<LogInformationAggregator*>(m_aggregator);
		if ( aggL ) return logGetSelectedNames();
	return out;
}


 std::vector<int> ManagerFileSelectorWidget::logGetSelectedIndexes() const
 {
	 std::vector<int> out;
	 QTreeWidget *treeWidget = m_treeWidget->getTreeWidget();
	 LogInformationAggregator *agg = dynamic_cast<LogInformationAggregator*>(m_aggregator);
	 if ( !agg ) return out;
	 for( int i = 0; i < treeWidget->topLevelItemCount(); ++i )
	 {
		 QTreeWidgetItem *item = treeWidget->topLevelItem(i);
		 if ( item->checkState(0) )
		 {
			 out.push_back(i);
		 }
	 }
	 return out;
 }


 std::vector<int> ManagerFileSelectorWidget::getSelectedIndexes() const{
	 std::vector<int> out;

	LogInformationAggregator *aggL = dynamic_cast<LogInformationAggregator*>(m_aggregator);
		if ( aggL ) return logGetSelectedIndexes();
	return out;
}

 void ManagerFileSelectorWidget::setFilterString(QString data)
 {
	 m_treeWidget->setFilterString(data);
 }
