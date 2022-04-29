'''Sql queries to create model and adam_opt tables'''

create_model_table = '''CREATE TABLE "Model" (
	"model_name"	TEXT NOT NULL,
	"type"	TEXT NOT NULL,
	"depth"	INTEGER,
	"input_shape"	TEXT,
	"optimizer_type"	TEXT,
	"optimizer_id"	INTEGER,
	"ensemble"	TEXT,
	"ensemble_type"	TEXT,
	"lc_config"	BLOB,
	PRIMARY KEY("model_name"),
	FOREIGN KEY("optimizer_id") REFERENCES "Adam_Opt"("optimizer_id")
);'''

create_adam_opt_table = '''CREATE TABLE "Adam_Opt" (
	"optimizer_id"	INTEGER NOT NULL UNIQUE,
	"learning_rate"	REAL NOT NULL,
	"amsgrad"	INTEGER NOT NULL,
	"decay"	REAL,
	"b1"	REAL,
	"b2"	REAL,
	"epsilon"	REAL,
	PRIMARY KEY("optimizer_id"),
	UNIQUE("learning_rate","amsgrad","decay","b1","b2","epsilon")
);'''

#cursor_obj.execute(create_model_table)