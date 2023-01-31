#include "stacksynchronizer.h"
#include "idata.h"
#include "folderdata.h"
#include "workingsetmanager.h"
#include <isohorizon.h>

#include <QComboBox>
#include <QListWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDialogButtonBox>
#include <QLabel>
#include <QCoreApplication>

#include <queue>

StackSynchronizer::StackSynchronizer(StackType stackType, QObject* parent) : QObject(parent) {
	m_stackType = stackType;
	m_stackClassType = StackClassType::NOTYPE;
}

StackSynchronizer::~StackSynchronizer() {
	clear();
}

bool StackSynchronizer::containsData(StackableData* data) const {
	std::vector<StackableDataHolder>::const_iterator it = std::find_if(m_datas.begin(), m_datas.end(),
			[data](const StackableDataHolder& holder) {
		return data==holder.data;
	});
	return it!=m_datas.end();
}

bool StackSynchronizer::addData(StackableData* data) {
	std::vector<StackableDataHolder>::const_iterator it = std::find_if(m_datas.begin(), m_datas.end(),
			[data](const StackableDataHolder& holder) {
		return data==holder.data;
	});
	bool ok = it==m_datas.end();

	if (ok) {
		std::shared_ptr<AbstractStack> stack = data->stack(m_stackType);

		if (m_datas.size()==0) {
			m_stackClassType = stack->stackClassType();
		} else {
			ok = m_stackClassType == stack->stackClassType();
		}

		if (ok) {
			StackableDataHolder holder;
			holder.data = data;
			holder.stack = stack;
			AbstractStack* stackPtr = stack.get();

			holder.syncConn = connect(holder.stack.get(), &AbstractStack::stackIndexChanged,
					[this, data, stackPtr](long index) {
				this->synchronizeStack(index, data, stackPtr);
			});

			// may not be the best way to prevent error linked to StackableData destruction
			if (QObject* object = dynamic_cast<QObject*>(data)) {
				holder.destroyConn = connect(object, &QObject::destroyed, [this, data]() {
					this->removeData(data);
				});
			}

			m_datas.push_back(holder);
		}
	}
	return ok;
}

bool StackSynchronizer::removeData(StackableData* data) {
	std::vector<StackableDataHolder>::const_iterator it = std::find_if(m_datas.begin(), m_datas.end(),
			[data](const StackableDataHolder& holder) {
		return data==holder.data;
	});
	bool ok = it!=m_datas.end();

	if (ok) {
		disconnect(it->syncConn);
		disconnect(it->destroyConn);
		m_datas.erase(it);
	}
	return ok;
}

StackType StackSynchronizer::stackType() const {
	return m_stackType;
}

StackClassType StackSynchronizer::stackClassType() const {
	return m_stackClassType;
}

void StackSynchronizer::synchronizeStack(long index, StackableData* data, AbstractStack* stack) {
	if (m_mainLock.tryLock()) {
		{
			QMutexLocker locker(&m_secondaryLock);
			m_nextIndex = -1;
			m_nextData = nullptr;
			m_nextStack = nullptr;
		}
		bool goOn = true;
		long nextIndex = index;
		StackableData* nextData = data;
		AbstractStack* nextStack = stack;
		while(goOn) {
			std::map<AbstractStack*, long> cache = synchronizeStackInternal(nextIndex, nextData, nextStack);
			QCoreApplication::processEvents();
			QMutexLocker locker(&m_secondaryLock);
			goOn = m_nextIndex>=0;
			if (goOn) {
				std::map<AbstractStack*, long>::const_iterator it =
						std::find_if(cache.begin(), cache.end(),
								[nextStack](const std::pair<AbstractStack*, long>& item) {
									return item.first==nextStack;
				});
				goOn = it==cache.end() || cache[nextStack]!=nextIndex;
			}
			if (goOn) {
				nextIndex = m_nextIndex;
				nextData = m_nextData;
				nextStack = m_nextStack;
				m_nextData = nullptr;
				m_nextStack = nullptr;
				m_nextIndex = -1;
			}
		}
		m_mainLock.unlock();
	} else {
		QMutexLocker locker(&m_secondaryLock);
		m_nextIndex = index;
		m_nextData = data;
		m_nextStack = stack;
	}
}

std::map<AbstractStack*, long> StackSynchronizer::synchronizeStackInternal(long index, StackableData* data, AbstractStack* stack) {
	std::map<AbstractStack*, long> cacheMap;
	if (m_stackClassType==StackClassType::RANGE) {
		AbstractRangeStack* rangeStack = dynamic_cast<AbstractRangeStack*>(stack);
		double value = rangeStack->stackValueFromIndex(index);

		// init cache for caller
		cacheMap[stack] = index;
		for (long i=0; i<m_datas.size(); i++) {
			if (m_datas[i].data!=data) {
				AbstractRangeStack* localRangeStack = dynamic_cast<AbstractRangeStack*>(m_datas[i].stack.get());
				StackableData* localData = m_datas[i].data;
				long localIndex = localRangeStack->stackIndexFromValue(value);

				cacheMap[m_datas[i].stack.get()] = localIndex;

				disconnect(m_datas[i].syncConn); // may be improved
				localRangeStack->setStackIndex(localIndex);
				// redo connection
				m_datas[i].syncConn = connect(localRangeStack, &AbstractStack::stackIndexChanged,
						[this, localData, localRangeStack](long index) {
					this->synchronizeStack(index, localData, localRangeStack);
				});
			}
		}
	}
	return cacheMap;
}

void StackSynchronizer::clear() {
	while (m_datas.size()>0) {
		StackableData* lastData = m_datas[m_datas.size()-1].data;
		removeData(lastData);
	}
}

StackSynchronizerDialog::StackSynchronizerDialog(WorkingSetManager* manager,
		QWidget* parent, Qt::WindowFlags f) : QDialog(parent, f) {
	m_manager = manager;
	m_synchronizer = nullptr;

	QVBoxLayout* mainLayout = new QVBoxLayout;
	setLayout(mainLayout);

	QHBoxLayout* typeLayout = new QHBoxLayout;
	mainLayout->addLayout(typeLayout);

	// default type is ISO
	m_currentType = StackType::ISO;
	typeLayout->addWidget(new QLabel("Type : "));
	m_typeComboBox = new QComboBox;
	m_typeComboBox->addItem("Iso", static_cast<int>(StackType::ISO));
	m_typeComboBox->addItem("Channel", static_cast<int>(StackType::CHANNEL));
	typeLayout->addWidget(m_typeComboBox);

	m_dataListWidget = new QListWidget;
	mainLayout->addWidget(m_dataListWidget);

	refreshDataList();

	QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
	mainLayout->addWidget(buttonBox);

	connect(buttonBox, &QDialogButtonBox::accepted, this, &StackSynchronizerDialog::accept);
	connect(buttonBox, &QDialogButtonBox::rejected, this, &StackSynchronizerDialog::reject);

	connect(m_typeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &StackSynchronizerDialog::stackTypeIndexChanged);
}

StackSynchronizerDialog::~StackSynchronizerDialog() {

}

void StackSynchronizerDialog::stackTypeIndexChanged(int index) {
	bool ok;
	int stackType = m_typeComboBox->itemData(index, Qt::UserRole).toInt(&ok);

	if (ok) {
		if (stackType==static_cast<int>(StackType::ISO)) {
			m_currentType = StackType::ISO;
		} else if (stackType==static_cast<int>(StackType::CHANNEL)) {
			m_currentType = StackType::CHANNEL;
		}

		refreshDataList();
	}
}

// caller take ownership of synchronizer
StackSynchronizer* StackSynchronizerDialog::newSynchronizer() {
	StackSynchronizer* synchronizer = new StackSynchronizer(m_currentType);

	initSynchronizer(synchronizer);

	return synchronizer;
}

// caller keep ownership of the synchronizer
void StackSynchronizerDialog::setSynchronizer(StackSynchronizer* synchronizer) {
	m_synchronizer = synchronizer;
	refreshDataList();
}

void StackSynchronizerDialog::refreshDataList() {
	m_dataListWidget->clear();

	std::queue<IData*> queue;

	// init queue
	QList<IData*> datas = m_manager->data();
	for (IData* data : datas) {
		queue.push(data);
	}

	while (queue.size()>0) {
		// FIFO
		IData* data = queue.front();
		queue.pop();
		// add new IData to queue
		if (FolderData* folder = dynamic_cast<FolderData*>(data)) {
			datas = folder->data();

			for (IData* data : datas) {
				queue.push(data);
			}
		}
		else if (IsoHorizon* horizon = dynamic_cast<IsoHorizon*>(data)) {
			for (IsoHorizon::Attribut attr : horizon->m_attribut) {
				if (attr.pFixedRGBLayersFromDatasetAndCube) {
					queue.push(attr.pFixedRGBLayersFromDatasetAndCube);
				} else if (attr.pFixedLayerImplIsoHorizonFromDatasetAndCube) {
					queue.push(attr.pFixedLayerImplIsoHorizonFromDatasetAndCube);
				}
			}
		} // other specific IData subclasses can contain IData, they can be added here


		//process current IData
		StackableData* stackableData = dynamic_cast<StackableData*>(data);
		if (stackableData) {
			std::vector<StackType> types = stackableData->stackTypes();
			bool typeValid = std::find(types.begin(), types.end(), m_currentType)!=types.end();
			if (typeValid) {
				QListWidgetItem* item = new QListWidgetItem;
				item->setText(data->name());
				item->setData(Qt::UserRole, QVariant::fromValue(data));
				item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
				if (m_synchronizer!=nullptr && m_currentType==m_synchronizer->stackType() &&
						m_synchronizer->containsData(stackableData)) {
					item->setCheckState(Qt::Checked);
				} else {
					item->setCheckState(Qt::Unchecked);
				}
				m_dataListWidget->addItem(item);
			}
		}
	}
}

void StackSynchronizerDialog::accept() {
	// update current synchronizer
	if (m_synchronizer!=nullptr) {
		initSynchronizer(m_synchronizer);
	}

	QDialog::accept();
}

void StackSynchronizerDialog::initSynchronizer(StackSynchronizer* synchronizer) {
	synchronizer->clear();

	long N = m_dataListWidget->count();
	for (long i=0; i<N; i++) {
		QListWidgetItem* item = m_dataListWidget->item(i);
		if (item->checkState()==Qt::Checked) {
			IData* data = qvariant_cast<IData*>(item->data(Qt::UserRole));
			StackableData* stackableData = dynamic_cast<StackableData*>(data);
			if (stackableData!=nullptr) {
				synchronizer->addData(stackableData);
			}
		}
	}
}
