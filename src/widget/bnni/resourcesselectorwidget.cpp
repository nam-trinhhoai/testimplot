#include "resourcesselectorwidget.h"

#include <QScrollArea>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QSpacerItem>
#include <QPushButton>
#include <QLabel>
#include <QCheckBox>
#include <QInputDialog>

ResourcesSelectorWidget::ResourcesSelectorWidget(QWidget* parent, Qt::WindowFlags f) :
		QWidget(parent, f) {
	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	m_scrollArea = new QScrollArea;
	m_resourceHolder = new QWidget;
	QVBoxLayout* holderLayout = new QVBoxLayout;
	m_resourceHolder->setLayout(holderLayout);
	m_resourcesLayout = new QGridLayout;
	holderLayout->addLayout(m_resourcesLayout);
	holderLayout->addItem(new QSpacerItem(0, 0, QSizePolicy::Minimum, QSizePolicy::Expanding));
	m_scrollArea->setWidget(m_resourceHolder);
	m_scrollArea->setWidgetResizable(true);
	mainLayout->addWidget(m_scrollArea);

	QPushButton* addResourceButton = new QPushButton("Add new computer");
	mainLayout->addWidget(addResourceButton);

	QPushButton* addCurrentButton = new QPushButton("Add current computer");
	mainLayout->addWidget(addCurrentButton);

	connect(addResourceButton, &QPushButton::clicked, this, &ResourcesSelectorWidget::addNewResource);
	connect(addCurrentButton, &QPushButton::clicked, this, &ResourcesSelectorWidget::addCurrentResource);
	connect(&m_resources, &ComputerResources::resourceAdded, this, &ResourcesSelectorWidget::resourceAdded);
	connect(&m_resources, &ComputerResources::resourceRemoved, this, &ResourcesSelectorWidget::resourceRemoved);

	addCurrentButton->click();
}

ResourcesSelectorWidget::~ResourcesSelectorWidget() {

}

ComputerResources& ResourcesSelectorWidget::resource() {
	return m_resources;
}

void ResourcesSelectorWidget::addNewResource() {
	QString hostName = QInputDialog::getText(this, tr("Computer Resource"), tr("New computer"));

	if (!hostName.isNull() && !hostName.isEmpty()) {
		ComputerResource* resource = new ComputerResource(hostName);
		bool valid = m_resources.addResource(resource);
		if (!valid) {
			resource->deleteLater();
		}
	}
}

void ResourcesSelectorWidget::addCurrentResource() {
	ComputerResource* resource = ComputerResource::getCurrentComputer();
	bool valid = m_resources.addResource(resource);
	if (!valid) {
		resource->deleteLater();
	}
}

void ResourcesSelectorWidget::resourceAdded(ComputerResource* resource) {
	long line = m_nextLine++;
	ResourceUI ui;
	ui.useCheckBox = new QCheckBox();
	ui.useCheckBox->setCheckState((resource->isAvailable()) ? Qt::Checked: Qt::Unchecked);
	ui.hostNameLabel = new QLabel(resource->hostName());

	m_resourcesLayout->addWidget(ui.useCheckBox, line, 0);
	m_resourcesLayout->addWidget(ui.hostNameLabel, line, 1);

	m_line2ItemMap[line] = ui;
	m_line2ResourceMap[line] = resource;
	m_line2UseMap[line] = resource->isAvailable();

	connect(ui.useCheckBox, &QCheckBox::stateChanged, [this, line](int state) {
		m_line2UseMap[line] = state==Qt::Checked;
	});
}

void ResourcesSelectorWidget::resourceRemoved(ComputerResource* resource) {
	std::map<long, ComputerResource*>::const_iterator it = std::find_if(m_line2ResourceMap.begin(),
			m_line2ResourceMap.end(), [resource](const std::pair<long, ComputerResource*>& pair) {
		return resource==pair.second;
	});

	if (it!=m_line2ResourceMap.end()) {
		long line = it->first;

		ResourceUI ui = m_line2ItemMap[line];
		m_line2ItemMap.erase(line);
		m_line2ResourceMap.erase(line);
		m_line2UseMap.erase(line);
		ui.useCheckBox->deleteLater();
		ui.hostNameLabel->deleteLater();
	}
}

std::vector<ComputerResource*> ResourcesSelectorWidget::getSelectedResources() const {
	std::vector<ComputerResource*> selectedResources;

	std::map<long, ComputerResource*>::const_iterator resourceIt = m_line2ResourceMap.begin();
	while (resourceIt!=m_line2ResourceMap.end()) {
		if (m_line2UseMap.at(resourceIt->first)) {
			selectedResources.push_back(resourceIt->second);
		}
		resourceIt++;
	}
	return selectedResources;

}

