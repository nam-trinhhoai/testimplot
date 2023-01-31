#ifndef DATAMANAGER_CONTROLER_FILESTORAGECONTROLER_H_
#define DATAMANAGER_CONTROLER_FILESTORAGECONTROLER_H_

#include <QObject>
#include <cstddef>
#include <map>

class LeafContainer;

class FileStorageControler : public QObject {
	Q_OBJECT
public:
	typedef struct ContainerDuo {
		LeafContainer* main = nullptr;
		LeafContainer* trash = nullptr;
	} ContainerDuo;

	FileStorageControler(QObject* parent=nullptr);
	~FileStorageControler();

	std::size_t addContainerDuo(ContainerDuo duo);
	bool removeContainerDuo(std::size_t id);

	bool removeLeafFromMainContainer(std::size_t leafKey, std::size_t containerKey);
	bool restoreLeafFromTrashContainer(std::size_t leafKey, std::size_t containerKey);
	bool restoreLeafFromTrashContainer(std::size_t leafKey, LeafContainer* trashContainer);// will search for the key then launch function above
	bool deleteLeafFromTrashContainer(std::size_t leafKey, std::size_t containerKey);
	bool deleteLeafFromTrashContainer(std::size_t leafKey, LeafContainer* trashContainer);// will search for the key then launch function above

	QString logPath() const;
	void setProjectPath(QString projectPath);
	void clearProjectPath();

private:
	std::size_t nextIndex() const;

	std::map<std::size_t, ContainerDuo> m_containers;
	std::map<std::size_t, QList<QMetaObject::Connection>> m_containerConnections;

	mutable std::size_t m_nextIndex = 0;
	QString m_logPath;
	QString m_projectPath;
	bool m_projectPathSet = false;
};

#endif
