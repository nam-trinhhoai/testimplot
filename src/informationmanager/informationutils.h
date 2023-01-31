#ifndef SRC_INFORMATIONMANAGER_INFORMATIONUTILS
#define SRC_INFORMATIONMANAGER_INFORMATIONUTILS

#include <QObject>
/**
 * did not manage to make Q_NAMESPACE work for information, there was an issue at link time
 * [100%] Linking CXX executable NextVisionViewer
 * libNextVisionViewerLib.so: undefined reference to `information::staticMetaObject'
 */
namespace information {
	enum class Property {
		AXIS_TYPE, // give SampleUnit
		CREATION_DATE, // give QList<QDateTime>, converted to QList<QDate> for filtering, but it is QList<QDatetime> for sorting
		MODIFICATION_DATE, // give QList<QDateTime>, converted to QList<QDate> for filtering, but it is QList<QDatetime> for sorting
		DATA_TYPE, // give SampleType
		NAME, // give QString
		OWNER, // give QStringList
		STORAGE_TYPE, // give StorageType
		VOLUME_TYPE, // give CUBE_TYPE
		WELL_BORE, // give QString
		WELL_HEAD, // give QString
		WELL_KIND, // give QStringList
		WELL_LOG_NAME,  // give QStringList
		WELL_PICK_NAME,  // give QStringList
		WELL_TFP_NAME // give QStringList
	};

	enum class StorageType {
		NEXTVISION,
		SISMAGE
	};
}; // namespace information

Q_DECLARE_METATYPE(information::Property)
Q_DECLARE_METATYPE(information::StorageType)

#endif // SRC_INFORMATIONMANAGER_INFORMATIONUTILS
