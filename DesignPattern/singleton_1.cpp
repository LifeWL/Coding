//懒汉
class Singleton{
private:
    static Singleton * instance;
    Singleton(){}
    Singleton(const Singleton& tmp){}
    Singleton& operator=(const Singleton& tmp){}
public:
    static Singleton* getInstance(){
        if(instance == NULL){
            instance = new Singleton();
        }
        return instance;
    }
};

//饿汉
Singleton* Singleton::instance = NULL;

class Singleton{
private:
    static pthread_mutex_t mutex;
    static Singleton * instance;
    Singleton(){
        pthread_mutex_init(&mutex, NULL); 
    }
    Singleton(const Singleton& tmp){}
    Singleton& operator=(const Singleton& tmp){}
public:
    static Singleton* getInstance(){
        pthread_mutex_lock(&mutex);
        if(instance == NULL){            
            instance = new Singleton();            
        }
        pthread_mutex_unlock(&mutex);
        return instance;
    }
};
Singleton* Singleton::instance = NULL;
pthread_mutex_t Singleton::mutex;

//懒汉
class Singleton{
private:
    static pthread_mutex_t mutex;
    Singleton(){
        pthread_mutex_init(&mutex, NULL);
    }
    Singleton(const Singleton& temp){}
    Singleton& operator=(const Singleton& temp){}
public:
    static Singleton* getInstance(){
        static Singleton instance;
        return &instance;
    }
};
pthread_mutex_t Singleton::mutex;

//饿汉
class Singleton{
private:
    static Singleton* instance;
    Singleton(const Singleton& temp){}
    Singleton& operator=(const Singleton& temp){}
protected:
	 Singleton(){} 
public:
    static Singleton* getInstance(){ 
        return instance;    
    }
};
Singleton* Singleton::instance = new Singleton();
