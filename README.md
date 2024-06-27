# rec-algorithm
This is for local dev environment.  
The distributed version will be add in the future,   
i am too busy working...

## recall
support i2i, hot, new and embedding.

## rank
LR

# data structure

## item

| name        | type   | required | description                          |
|-------------|--------|----------|--------------------------------------|  
| id          | string | yes      | item uniq id                         |  
| title       | string | yes      |                                      |
| category    | string | yes      | single value                         |  
| tags        | string | no       | multi value                          |  
| scene       | string | yes      | relation recommend or guess you like |  
| pub_time    | int    | yes      |                                      |  
| modify_time | int    | no       | update time                          |  
| expire_time | int    | no       |                                      |  
| status      | bool   | yes      | could be recommend                   |  
| weight      | int    | no       |                                      |  
| ext_fields  | json   | no       |                                      |  

## user

| name          | type   | required | description    |
|---------------|--------|----------|----------------|  
| id            | string | yes      | user uniq id   |  
| device_id     | string | yes      | user device id |
| name          | string | no       | fake name      |  
| gender        | string | no       |                |  
| age           | int    | no       |                |  
| country       | string | no       |                |  
| city          | string | no       | update time    |  
| phone         | long   | no       |                |  
| tags          | string | no       | multi value    |  
| register_time | int    | no       |                |
| login_time    | int    | no       |                |
| ext_fields    | json   | no       |                |

## event

| name       | type   | required | description                            |
|------------|--------|----------|----------------------------------------|  
| id         | string | yes      | event uniq id                          |  
| user_id    | string | yes      |                                        |
| item_id    | string | yes      | single value                           |  
| trace_id   | string | yes      | eg: openrec                            |  
| scene      | string | yes      | relation recommend or guess you like   |  
| type       | string | yes      | eg: click, expose, buy, collect, stay  |  
| value      | string | yes      | the value of the type                  |  
| time       | int    | yes      |                                        |  
| is_login   | bool   | no       | could be recommend                     |  
| ext_fields | json   | no       |                                        |  

